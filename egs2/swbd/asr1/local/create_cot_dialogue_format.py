import sys
import os


def process_text(line):
    """Process text by extracting content after <TEXT> tags and joining."""
    line1 = line.split(" <UTT> ")
    line2 = [k.split(" <TEXT> ")[-1].strip() for k in line1]
    return " ".join(line2).strip()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_dir> <output_dir>")
        print("  Example: python script.py data/train_fisher data/train_fisher_response")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Input files
    segment_file_path = os.path.join(input_dir, "segments")
    text_file_path = os.path.join(input_dir, "text.punctuate")
    
    # Output files
    segments_output = os.path.join(output_dir, "segments")
    src_text_output = os.path.join(output_dir, "src_text")
    response_text_output = os.path.join(output_dir, "text")
    tgt_segments_output = os.path.join(output_dir, "tgt_segments")
    
    # Read input files
    with open(segment_file_path, "r") as segment_file:
        segment_line_arr = [line for line in segment_file]
    
    with open(text_file_path, "r") as text_file:
        text_line_arr = [line for line in text_file]
    
    # Build text dictionary
    text_dict = {}
    for line in text_line_arr:
        text_dict[line.split()[0]] = 1
    
    # Filter segments based on text dictionary
    segment_line_arr1 = []
    for k in segment_line_arr:
        if k.split()[0] in text_dict:
            segment_line_arr1.append(k)
    
    segment_line_arr = segment_line_arr1
    
    # Process segments
    segment_dict = {}
    map_dict = {}
    remove_dialog_id_dict = {}
    
    for line_count in range(len(segment_line_arr)):
        segment_line = segment_line_arr[line_count]
        text_line = text_line_arr[line_count]
        
        if len(text_line.strip().split()[1:]) < 5:
            continue
        
        assert segment_line.split()[0] == text_line.split()[0]
        text = " ".join(text_line.strip().split()[1:])
        
        if "<UTT>" in text:
            import pdb; pdb.set_trace()
        
        segment_line1 = segment_line.strip().split()
        dialog_id = segment_line1[1].split("-")[0]
        speaker_id = segment_line1[1].split("-")[-1]
        start_time = float(segment_line1[2])
        end_time = float(segment_line1[3])
        
        if end_time - start_time > 30:
            print(segment_line)
            end_time = start_time + 29
            if dialog_id not in remove_dialog_id_dict:
                remove_dialog_id_dict[dialog_id] = 1
        
        if dialog_id not in segment_dict:
            segment_dict[dialog_id] = {}
        
        if start_time not in segment_dict[dialog_id]:
            segment_dict[dialog_id][start_time] = [speaker_id, end_time, text]
        
        map_dict["-".join([dialog_id, str(start_time), speaker_id, str(end_time)])] = text_line.strip().split()[0]
    
    # Write output files
    with open(segments_output, "w") as file_write, \
         open(src_text_output, "w") as text_file_write, \
         open(response_text_output, "w") as response_text_file_write, \
         open(tgt_segments_output, "w") as response_file_write:
        
        for dialog_id in segment_dict:
            if dialog_id in remove_dialog_id_dict:
                continue
            
            sorted_dict = {key: segment_dict[dialog_id][key] for key in sorted(segment_dict[dialog_id])}
            prev_speaker = None
            prev_end_time = None
            sorted_joined_arr = []
            
            for start_time in sorted_dict:
                speaker = sorted_dict[start_time][0]
                
                if prev_speaker is not None:
                    assert prev_speaker == sorted_joined_arr[-1][0]
                
                if prev_speaker is None:
                    sorted_joined_arr.append([speaker, start_time] + sorted_dict[start_time][1:])
                    prev_speaker = speaker
                elif speaker != prev_speaker:
                    if sorted_dict[start_time][1] > sorted_joined_arr[-1][-2]:
                        sorted_joined_arr.append([speaker, start_time] + sorted_dict[start_time][1:])
                        prev_speaker = speaker
                elif speaker == prev_speaker:
                    if sorted_dict[start_time][1] > sorted_joined_arr[-1][-2]:
                        sorted_joined_arr[-1][-1] = sorted_joined_arr[-1][-1] + " <UTT> " + str(start_time) + " " + str(sorted_joined_arr[-1][-2]) + " <TEXT> " + sorted_dict[start_time][-1]
                        sorted_joined_arr[-1][-2] = sorted_dict[start_time][1]
                    else:
                        continue
            
            for count in range(len(sorted_joined_arr) - 1):
                k = sorted_joined_arr[count]
                
                if k[2] - k[1] > 30:
                    start_time = k[1]
                    while k[2] - start_time > 30:
                        utt_arr = k[-1].split(" <UTT> ")
                        line1 = " <UTT> ".join(utt_arr[1:])
                        start_time = float(line1.split(" <TEXT> ")[0].split()[0])
                        line1 = " <TEXT> ".join(line1.split(" <TEXT> ")[1:])
                        k[1] = start_time
                        k[-1] = line1
                
                id1 = dialog_id + "-" + k[0] + "_" + "{:06}".format(int(k[1] * 100)) + "-" + "{:06}".format(int(k[2] * 100))
                file_write.write(id1 + " " + dialog_id + "-" + k[0] + " " + str(k[1]) + " " + str(k[2]) + "\n")
                text_file_write.write(id1 + " " + process_text(k[-1]) + "\n")
                
                next_k = sorted_joined_arr[count + 1].copy()
                
                if (next_k[2] - next_k[1] > 30):
                    end_time = next_k[2]
                    while end_time - next_k[1] > 30:
                        if " <UTT> " not in next_k[-1]:
                            import pdb; pdb.set_trace()
                        else:
                            utt_arr = next_k[-1].split(" <UTT> ")
                            line1 = " <UTT> ".join(utt_arr[:-1])
                            end_time = float(utt_arr[-1].split(" <TEXT> ")[0].split()[-1])
                            next_k[2] = end_time
                            next_k[-1] = line1
                
                response_file_write.write(id1 + " " + dialog_id + "-" + next_k[0] + " " + str(next_k[1]) + " " + str(next_k[2]) + "\n")
                response_text_file_write.write(id1 + " " + process_text(next_k[-1]) + "\n")
    
    print(f"Processing complete including removing backchannels.")
    print(f"Output files written to:")
    print(f"  - {segments_output}")
    print(f"  - {src_text_output}")
    print(f"  - {response_text_output}")
    print(f"  - {tgt_segments_output}")