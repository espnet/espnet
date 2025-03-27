# adapted from Koenecke et al (2020)
# https://github.com/stanford-policylab/asr-disparities/blob/master/
#   src/utils/snippet_generation.py
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from pydub import AudioSegment


def load_coraal_text(project_path):
    # Load CORAAL transcripts and metadata files
    #   which are assumed to be in project_path without nesting
    file_pattern = os.path.join(project_path, "*metadata*.txt")
    filenames = glob.glob(file_pattern)
    print("metadata files", filenames)
    metadata = pd.concat(
        [pd.read_csv(filename, sep="\t") for filename in filenames], sort=False
    )

    rows = []
    for sub, file in metadata[["CORAAL.Sub", "CORAAL.File"]].drop_duplicates().values:
        if file == "VLD_se0_ag2_f_01_2":
            # this file is missing from VLD/2021.07 (2023.06 release)
            continue

        text_filename = os.path.join(project_path, file + ".txt")

        text = pd.read_csv(text_filename, sep="\t")
        # columns: Line, Spkr, StTime, Content, EnTime
        # each row tends to be short

        # filter out rows with pauses
        text["pause"] = text.Content.str.contains(r"(pause \d+(\.\d{1,2})?)")
        text = text[~text.pause]

        for spkr, sttime, content, entime in text[
            ["Spkr", "StTime", "Content", "EnTime"]
        ].values:
            row = {
                "name": spkr,
                "speaker": spkr,
                "start_time": sttime,
                "end_time": entime,
                "content": content,
                "filename": text_filename,
                "source": "coraal",
                "location": sub,
                "basefile": file,
                "interviewee": spkr in file,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["basefile", "start_time"])
    df["line"] = np.arange(len(df))
    df = df[
        [
            "basefile",
            "line",
            "start_time",
            "end_time",
            "speaker",
            "content",
            "interviewee",
            "source",
            "location",
            "name",
            "filename",
        ]
    ]
    print("CORAAL full df len ", len(df))
    df.drop_duplicates()
    print("CORAAL dedup df len ", len(df))
    full_df = df.merge(
        metadata,
        left_on=["basefile", "speaker"],
        right_on=["CORAAL.File", "CORAAL.Spkr"],
        how="left",
    )
    print("CORAAL full merged metadata len ", len(full_df))
    return full_df


def find_snippet(snippets, basefile, start_time, end_time):
    # Sanity check: given a start and end time of speech in transcript
    #   confirm the corresponding snippet exists
    start_time = round(start_time, 3)
    end_time = round(end_time, 3)
    match = snippets[
        (snippets.basefile == basefile)
        & (round(snippets.start_time, 3) == start_time)
        & (round(snippets.end_time, 3) == end_time)
    ]
    if len(match) == 0:
        print(
            "Snippet not found at {} from {} to {}".format(
                basefile, start_time, end_time
            )
        )
    return match


def segment_filename(basefilename, start_time, end_time, buffer_val):
    # Generate new filename for snippet subsets of .wav files
    #   (including start and end times)
    start_time = int((start_time - buffer_val) * 1000)  # ms
    end_time = int((end_time + buffer_val) * 1000)  # ms
    return f"{basefilename}_{start_time:012d}_{end_time:012d}"


def create_coraal_snippets(transcripts):
    # Make dataframe of snippets from transcripts, applying filters
    snippets = []

    for basefile in transcripts.basefile.unique():
        df = transcripts[transcripts.basefile == basefile][
            ["line", "start_time", "end_time", "interviewee", "content"]
        ]
        backward_check = df["start_time"].values[1:] >= df["end_time"].values[:-1]
        backward_check = np.insert(backward_check, 0, True)
        forward_check = df["end_time"].values[:-1] <= df["start_time"].values[1:]
        forward_check = np.insert(forward_check, len(forward_check), True)
        # filter out
        #   interviewer lines
        #   overlapping speech []
        #   non-linguistic content <>
        #   dysfluencies, redacted speech, inaudible speech //
        #   line-level notes ()
        df["use"] = (
            backward_check
            & forward_check
            & df.interviewee
            & ~df.content.str.contains(r"\[")
            & ~df.content.str.contains("]")
            & ~df.content.str.contains("<")
            & ~df.content.str.contains(">")
            & ~df.content.str.contains("/")
            & ~df.content.str.contains(r"\(")
            & ~df.content.str.contains(r"\)")
        )

        values = df[["line", "use"]].values
        snippet = []
        for i in range(len(values)):
            line, use = values[i]
            # start a new snippet
            if use:
                snippet.append(line)
            elif snippet:
                # if shouldn't use this line, but snippet exists
                # save previous snippet, start a new snippet
                snippets.append(snippet)
                snippet = []
            # IMPROVEMENT: segment the file if it's too long
            # see https://github.com/cmu-llab/s3m-aave/blob/
            #       main/data/nsp/segment.py
        if snippet:
            snippets.append(snippet)

    basefiles = transcripts.basefile.values
    start_times = transcripts.start_time.values
    end_times = transcripts.end_time.values
    contents = transcripts.content.values
    # metadata comes from the metadata files
    #   which was merged in load_coraal_text
    gender = transcripts.Gender.values
    age = transcripts.Age.values
    age_group = transcripts["Age.Group"].values
    socioeconomic_group = transcripts["CORAAL.SEC.Group"].values
    education_group = transcripts["Edu.Group"].values
    speaker_id = transcripts["CORAAL.Spkr"].values
    location = transcripts["location"].values

    rows = []
    for indices in snippets:
        rows.append(
            {
                "basefile": basefiles[indices[0]],
                "start_time": start_times[indices[0]],
                "end_time": end_times[indices[-1]],
                "content": " ".join(contents[indices]),
                "age": age[indices[0]],
                "gender": gender[indices[0]],
                "age_group": age_group[indices[0]],
                "socioeconomic_group": socioeconomic_group[indices[0]],
                "education_group": education_group[indices[0]],
                "speaker_id": speaker_id[indices[0]],
                "location": location[indices[0]],
            }
        )
    snippets = pd.DataFrame(rows)[
        [
            "basefile",
            "start_time",
            "end_time",
            "content",
            "age",
            "gender",
            "age_group",
            "socioeconomic_group",
            "education_group",
            "speaker_id",
            "location",
        ]
    ]
    snippets = snippets.sort_values(["basefile", "start_time"])
    snippets["duration"] = snippets.end_time - snippets.start_time  # seconds
    snippets["segment_filename"] = [
        segment_filename(b, s, e, buffer_val=0)
        for b, s, e in snippets[["basefile", "start_time", "end_time"]].values
    ]
    return snippets


def get_nonexistent_snippets(input_folder, snippets):
    nonexistent_snippets = set()
    for group_key, group_df in snippets.groupby(["basefile"]):
        (basefile,) = group_key
        audio = AudioSegment.from_file(f"{input_folder}/{basefile}.wav")
        for _, snippet in group_df.iterrows():
            start_time, end_time, segment_filename = (
                snippet["start_time"],
                snippet["end_time"],
                snippet["segment_filename"],
            )

            if (
                audio[int(start_time * 1000) : int(end_time * 1000)].duration_seconds
                == 0.0
            ):
                nonexistent_snippets.add(segment_filename)

    return nonexistent_snippets


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Help: python local/snippet_generation.py <input_folder> "
            "<output_folder> <min_duration> <max_duration>"
        )
        print("ex: python local/snippet_generation.py " "downloads downloads 0.1 30")
        print(
            "Note: This script assumes all files (metadata and wav) "
            "are in <input_folder> with no nested folders"
        )
        exit(1)
    base_folder = sys.argv[1]
    MIN_DURATION = float(sys.argv[3])  # in seconds
    MAX_DURATION = float(sys.argv[4])  # in seconds
    output_folder = sys.argv[2]  # ex: 'downloads'
    start_time = datetime.now()

    # Create CORAAL snippets
    coraal_transcripts = load_coraal_text(base_folder)
    coraal_snippets = create_coraal_snippets(coraal_transcripts)

    # These snippets should exist, run these pre-filtering on duration
    assert (
        len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 364.6292, 382.2063)) > 0
    )
    assert (
        len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 17.0216, 19.5291)) > 0
    )
    assert (
        len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 875.0084, 876.5177)) > 0
    )
    assert (
        len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 885.9359, 886.3602)) > 0
    )
    assert (
        len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 890.9707, 894.35)) > 0
    )
    assert (
        len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 895.9076, 910.211)) > 0
    )

    # ensure no annotations left
    assert (
        len(
            coraal_snippets[
                (coraal_snippets.content.str.contains(r"\["))
                | (coraal_snippets.content.str.contains(r"\]"))
                | (coraal_snippets.content.str.contains("<"))
                | (coraal_snippets.content.str.contains(">"))
                | (coraal_snippets.content.str.contains("/"))
                | (coraal_snippets.content.str.contains(r"\("))
                | (coraal_snippets.content.str.contains(r"\)"))
            ]
        )
        == 0
    )
    interviewees = {
        b: s
        for b, s in coraal_transcripts[coraal_transcripts.interviewee][
            ["basefile", "speaker"]
        ]
        .drop_duplicates()
        .values
    }

    # Check for non-overlapping snippets
    for basefile, start_time, end_time in coraal_snippets[
        ["basefile", "start_time", "end_time"]
    ].values:
        xscript_speakers = coraal_transcripts[
            (coraal_transcripts.basefile == basefile)
            & (coraal_transcripts.start_time >= start_time)
            & (coraal_transcripts.end_time <= end_time)
        ].speaker.unique()
        if len(xscript_speakers) < 1:
            print(basefile, start_time, end_time, "not enough speakers")
            assert 0
        if not (
            len(xscript_speakers) == 1 and xscript_speakers[0] == interviewees[basefile]
        ):
            print(basefile, start_time, end_time, "interviewee missing")
            assert 0

    # Restrict to only snippets of specified duration range (e.g. 0.1 - 30s)
    coraal_snippets = coraal_snippets[
        (MIN_DURATION <= coraal_snippets.duration)
        & (coraal_snippets.duration <= MAX_DURATION)
    ]

    nonexistent_snippets = get_nonexistent_snippets(base_folder, coraal_snippets)
    print(nonexistent_snippets)
    coraal_snippets = coraal_snippets[
        ~coraal_snippets["segment_filename"].isin(nonexistent_snippets)
    ]

    # save transcripts (which includes speaker metadata)
    print(coraal_snippets.duration.describe())
    coraal_snippets.to_csv(output_folder + "/transcript.tsv", sep="\t", index=False)

    # generate segments file (kaldi)
    # <utterance_id> <wav_id> <start_time> <end_time>
    coraal_snippets[["segment_filename", "basefile", "start_time", "end_time"]].to_csv(
        output_folder + "/segments", sep=" ", header=False, index=False
    )

    print("finished in", datetime.now() - start_time)
