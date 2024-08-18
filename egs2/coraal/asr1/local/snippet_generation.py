# adapted from Koenecke et al (2020)
# https://github.com/stanford-policylab/asr-disparities/blob/master/src/utils/snippet_generation.py
import os
import glob
import sys
import pandas as pd
import numpy as np
from pydub import AudioSegment

def load_coraal_text(project_path):
    # Load CORAAL transcripts where the project path for each location contains that location's metadata file,
    # as well as subdirectories 'audio' and 'transcripts' (containing all .wav and .txt files, respectively)
    file_pattern = os.path.join(project_path, '*/*metadata*.txt')
    filenames = glob.glob(file_pattern, recursive=True)
    print(filenames)
    metadata = pd.concat([pd.read_csv(filename, sep='\t') for filename in filenames], sort=False)
    
    rows = []
    
    for sub, file in metadata[['CORAAL.Sub', 'CORAAL.File']].drop_duplicates().values:
        subpath = os.path.join(project_path, sub.lower())
        text_filename = os.path.join(subpath, 'transcripts', file + '.txt')

        text = pd.read_csv(text_filename, sep='\t')
        text['pause'] = text.Content.str.contains('(pause \d+(\.\d{1,2})?)')
        text = text[~text.pause]
        
        for spkr, sttime, content, entime in text[['Spkr', 'StTime', 'Content', 'EnTime']].values:
            row = {
                'name': spkr,
                'speaker': spkr,
                'start_time': sttime,
                'end_time': entime,
                'content': content,
                'filename': text_filename,
                'source': 'coraal',
                'location': sub,
                'basefile': file,
                'interviewee': spkr in file
            }
            rows.append(row)
        
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['basefile', 'start_time'])
    df['line'] = np.arange(len(df))
    df = df[['basefile', 'line', 'start_time', 'end_time', 'speaker', 'content', 'interviewee', 'source', 'location', 'name', 'filename']]
    print("CORAAL full df len ", len(df))
    df.drop_duplicates()
    print("CORAAL dedup df len ", len(df))
    full_df = df.merge(metadata, left_on=['basefile', 'speaker'], right_on=['CORAAL.File', 'CORAAL.Spkr'], how = 'left')
    print("CORAAL full merged metadata len ", len(full_df))
    return full_df

def find_snippet(snippets, basefile, start_time, end_time):
    # Sanity check: given a start and end time of speech in transcript, confirm the corresponding snippet exists
    start_time = round(start_time, 3)
    end_time = round(end_time, 3)
    match = snippets[(snippets.basefile == basefile) & (round(snippets.start_time, 3) == start_time) & (round(snippets.end_time, 3) == end_time)]
    if len(match) == 0:
        print("Snippet not found at {} from {} to {}".format(basefile, start_time, end_time))
    return match

def segment_filename(basefilename, start_time, end_time, buffer_val):
    # Generate new filename for snippet subsets of .wav files (including start and end times)
    start_time = int((start_time-buffer_val)*1000)
    end_time = int((end_time+buffer_val)*1000)
    filename = "{}_{}_{}.wav".format(basefilename, start_time, end_time)
    return filename

def create_segment(src_file, dst_dir, basefilename, start_time, end_time, buffer_val):
    # Construct snippet segmentation of .wav files
    segment_basename = segment_filename(basefilename, start_time, end_time, buffer_val)
    segment_file = os.path.join(dst_dir, segment_basename)
    if not os.path.isfile(src_file):
        print("Error: Source file {} not found".format(src_file))
        return
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    audio = AudioSegment.from_wav(src_file)
    start_time = int((start_time-buffer_val)*1000)
    end_time = int((end_time+buffer_val)*1000)
    clip = audio[start_time:end_time]
    clip.export(segment_file, format="wav")
    return segment_file

def create_coraal_snippets(transcripts):
    # Make dataframe of snippets from transcripts 
    snippets = []
    
    for basefile in transcripts.basefile.unique():
        df = transcripts[transcripts.basefile == basefile][['line', 'start_time', 'end_time', 'interviewee', 'content',
                                                           'Gender', 'Age']]
        backward_check = df['start_time'].values[1:] >= df['end_time'].values[:-1]
        backward_check = np.insert(backward_check, 0, True)
        forward_check = df['end_time'].values[:-1] <= df['start_time'].values[1:]
        forward_check = np.insert(forward_check, len(forward_check), True)
        df['use'] = backward_check & forward_check & df.interviewee \
            & ~df.content.str.contains('\[') \
            & ~df.content.str.contains(']')
        
        values = df[['line', 'use']].values
        snippet = []
        for i in range(len(values)):
            line, use = values[i]
            if use:
                snippet.append(line)
            elif snippet: # if shouldn't use this line, but snippet exists
                snippets.append(snippet)
                snippet = []
        if snippet:
            snippets.append(snippet)
            
    basefiles = transcripts.basefile.values
    start_times = transcripts.start_time.values
    end_times = transcripts.end_time.values
    contents = transcripts.content.values
    gender = transcripts.Gender.values
    age = transcripts.Age.values
    rows = []
    for indices in snippets:
        rows.append({
            'basefile': basefiles[indices[0]],
            'start_time': start_times[indices[0]],
            'end_time': end_times[indices[-1]],
            'content': ' '.join(contents[indices]),
            'age': age[indices[0]],
            'gender': gender[indices[0]]
        })
    snippets = pd.DataFrame(rows)[['basefile', 'start_time', 'end_time', 'content', 'age', 'gender']]
    snippets = snippets.sort_values(['basefile', 'start_time'])
    snippets['duration'] = snippets.end_time - snippets.start_time
    snippets['segment_filename'] = [segment_filename(b, s, e, buffer_val=0) for b, s, e in snippets[['basefile', 'start_time', 'end_time']].values]
    return snippets

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Command example: python3 snippet_generation.py <input_folder> <output_tsv>")
        exit(1)    
        
    base_folder = sys.argv[1]

    # Create CORAAL snippets
    coraal_transcripts = load_coraal_text(base_folder)
    coraal_snippets = create_coraal_snippets(coraal_transcripts)

    # These snippets should exist, run these pre-filtering on duration
    assert len(find_snippet(coraal_snippets, 'DCB_se1_ag1_f_01_1', 8.9467, 12.4571)) > 0
    assert len(find_snippet(coraal_snippets, 'DCB_se1_ag1_f_01_1', 364.6292, 382.2063)) > 0
    assert len(find_snippet(coraal_snippets, 'DCB_se1_ag1_f_01_1', 17.0216, 19.5291)) > 0
    assert len(find_snippet(coraal_snippets, 'DCB_se1_ag1_f_01_1', 875.0084, 876.5177)) > 0
    assert len(find_snippet(coraal_snippets, 'DCB_se1_ag1_f_01_1', 885.9359, 886.3602)) > 0
    assert len(find_snippet(coraal_snippets, 'DCB_se1_ag1_f_01_1', 890.9707, 894.35)) > 0
    assert len(find_snippet(coraal_snippets, 'DCB_se1_ag1_f_01_1', 895.9076, 910.211)) > 0

    assert len(coraal_snippets[(coraal_snippets.content.str.contains('\[')) | (coraal_snippets.content.str.contains('\]'))]) == 0
    interviewees = {b: s for b, s in coraal_transcripts[coraal_transcripts.interviewee][['basefile', 'speaker']].drop_duplicates().values}

    # Check for non-overlapping snippets
    for basefile, start_time, end_time in (coraal_snippets[['basefile', 'start_time', 'end_time']].values):
        xscript_speakers = coraal_transcripts[(coraal_transcripts.basefile == basefile)
                          & (coraal_transcripts.start_time >= start_time)
                          & (coraal_transcripts.end_time <= end_time)].speaker.unique()
        if len(xscript_speakers) < 1:
            print(basefile, start_time, end_time)
            assert 0
        if not (len(xscript_speakers) == 1 and xscript_speakers[0] == interviewees[basefile]):
            print(basefile, start_time, end_time)
            assert 0

        # Generate .wav files for each snippet from larger audio clips
        if basefile[0:3] == 'DCB':
            dst_dir = base_folder + '/dcb/audio_segments/'
        elif basefile[0:3] == 'PRV':
            dst_dir = base_folder + '/prv/audio_segments/'
        elif basefile[0:3] == 'ROC':
            dst_dir = base_folder + 'roc/audio_segments/'        
        src_file = base_folder + basefile[0:3].lower() + '/audio/' + basefile + '.wav'
        create_segment(src_file, dst_dir, basefile, start_time, end_time, 0)

    # Restrict to only 5-50 second snippets
    min_duration = 5 # in seconds
    max_duration = 50 # in seconds
    coraal_snippets = coraal_snippets[(min_duration <= coraal_snippets.duration) & (coraal_snippets.duration <= max_duration)]
    print(coraal_snippets.duration.describe())

    output_loc = sys.argv[2] #base_folder + 'input/coraal_snippets.tsv'
    coraal_snippets.to_csv(output_loc, sep = '\t', index = False)
