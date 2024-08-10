import argparse
import os


def find_audio_files(root_dir, extensions=(".flac", ".wav")):
    """
    Recursively finds all files with specified extensions in the given directory.

    Args:
        root_dir (str): The root directory to start the search from.
        extensions (tuple): File extensions to search for.

    Returns:
        list: A list of tuples containing file names and their absolute paths.
    """
    audio_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                absolute_path = os.path.abspath(os.path.join(dirpath, filename))
                audio_files.append((filename, absolute_path))
            elif filename.lower().endswith("mp3"):
                absolute_path = os.path.join(dirpath, filename)
                absolute_path = "ffmpeg -i {} -f wav -ar 44100 -ab 16 -ac 1 - |".format(
                    absolute_path
                )
                audio_files.append((filename, absolute_path))
    return audio_files


def document_audio_files(audio_files, output_file, key_prefix):
    """
    Documents the found audio files in a specified format.

    Args:
        audio_files (list): A list of tuples containing file names
            and their absolute paths.
        output_file (str): The file where the documentation will be written.
        key_prefix (str): The key prefix for wavid.
    """
    with open(output_file, "w") as f:
        for filename, path in audio_files:
            f.write(f"{key_prefix}_{filename} {path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=str)
    parser.add_argument("key_prefix", type=str)
    args = parser.parse_args()
    root_directory = args.root_directory
    output_file = "{}.scp".format(args.key_prefix)

    audio_files = find_audio_files(root_directory)
    document_audio_files(audio_files, output_file, args.key_prefix)

    print(f"Audio files have been documented in {output_file}")
