import os
import json
import argparse
import shutil


def draw_histogram(source_ls, prefix=''):
    import matplotlib.pyplot as plt
    plt.hist(source_ls, bins=50, edgecolor='black')

    plt.title(f"Histogram of {prefix} duration")
    plt.xlabel("Duration")
    plt.ylabel("Number of phn")

    plt.savefig(f"duration_histogram_{prefix}.png")


def process_line_label(line, visualize_distribution=False):
    parts = line.split()
    audio_id = parts[0]
    data_unit = parts[1:]
    
    duration_ls = []

    result = [audio_id]
    for i in range(0, len(data_unit), 3):
        start_time = float(data_unit[i])
        end_time = float(data_unit[i+1])
        phone = data_unit[i+2]
        
        duration = (end_time - start_time) * 1000 / 20
        duration_rounded = round(duration)

        if visualize_distribution:
            duration_ls.append(duration_rounded)
        
        result.append(f"{duration_rounded} svs_{phone}")
    

    return ' '.join(result), duration_ls


def process_line_score(line):
    parts = line.split()
    tempo = parts[0]
    data_unit = parts[1:]
    
    result = [tempo]
    for i in range(0, len(data_unit), 5):
        start_time = float(data_unit[i])
        end_time = float(data_unit[i+1])
        phn = str(data_unit[i+2])
        midi = str(data_unit[i+3])
        
        duration = (end_time - start_time) * 1000 / 20
        duration_rounded = str(round(duration))
        
        result.append(f"{duration_rounded} svs_{midi} svs_{phn}")
    
    return ' '.join(result)


def duration2repeat(data_folder):
    """(duration, phn, midi) --> (phn, midi)*duration, <placeholder>"""
    
    file_path = os.path.join(data_folder, 'label')
    file_name, file_extension = os.path.splitext(file_path)
    backup_file_name = f"{file_name}_format1{file_extension}"
    print(f"pre {file_path}, aft {backup_file_name}")
    shutil.copy(file_path, backup_file_name)

    file_out = os.path.join(data_folder, 'label2')

    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    with open(file_out, 'w') as outfile:
        for line in lines:
            parts = line.split()
            audio_id = parts[0]
            data_unit = parts[1:]
            
            result = [audio_id]
            for i in range(0, len(data_unit), 3):
                duration = int(data_unit[i])
                phn = str(data_unit[i+1])
                midi = str(data_unit[i+2])
                
                result.extend([f"{phn} {midi}"] * duration)
                result.append("<svs_placeholder>") # use a new token to define placeholder

            processed_line = ' '.join(result[:-1])
            outfile.write(processed_line + '\n')
    shutil.copy(file_out, file_path)

def overwrite_file(data_folder, file_type, visualize_distribution=False):
    file_path = os.path.join(args.data_folder, args.file_type)
    
    # backup the original file
    base_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(base_name)
    backup_file_name = f"{file_name}_backup{file_extension}"
    backup_file_path = os.path.join(args.data_folder, backup_file_name)
    shutil.copy(file_path, backup_file_path)

    with open(file_path, 'r') as infile:
        lines = infile.readlines()
    
    prefix = args.data_folder.split('/')[-1]
    print(f'Processsing dataset {prefix}')
    file_test = os.path.join(args.data_folder, file_type)

    if file_type=="label":
        if visualize_distribution:
            durations = []

        with open(file_test, 'w') as outfile:
            for line in lines:
                processed_line, duration_line = process_line_label(line.strip(), visualize_distribution)
                if visualize_distribution:
                    durations.extend(duration_line)
                outfile.write(processed_line + '\n')

        if visualize_distribution:
            # visualize phn duration distribution
            draw_histogram(durations, prefix)

    elif file_type=="score":
        with open(file_test, 'w') as outfile:
            for line in lines:
                processed_line = process_line_score(line.strip())
                outfile.write(processed_line + '\n')

    else:
        print(f"unsupported file type")



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="""
        Tokenize duration label by: (ed-st)*1k/20f.""")
    parser.add_argument("--data_folder", type = str, required = True,
                        help="in processing dataset")
    parser.add_argument("--file_type", type = str, required = True,
                        help="in processing file")
    args = parser.parse_args()

    if args.file_type=="label" or args.file_type=="score":
        overwrite_file(args.data_folder, args.file_type)

    elif args.file_type=="duration":
        duration2repeat(args.data_folder)
