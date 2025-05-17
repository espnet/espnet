import os
from typing import Dict, Union


def summarize_tree(
    path, max_dirs=5, max_files=5, sample_exts={".flac", ".mp3", ".wav", ".m4a"}
):
    def summarize_dir(current_path) -> Dict[str, Union[list, dict]]:
        summary = {}
        try:
            entries = os.listdir(current_path)
        except Exception as e:
            return {"_error": str(e)}
        dirs, bulk, meta = [], [], []
        for name in sorted(entries):
            full_path = os.path.join(current_path, name)
            if os.path.isdir(full_path):
                dirs.append(name)
            else:
                ext = os.path.splitext(name)[1]
                if ext in sample_exts:
                    bulk.append(name)
                else:
                    meta.append(name)
        # summarize subdirectories
        dir_summaries = {}
        for subdir in dirs[:max_dirs]:
            dir_summaries[subdir] = summarize_dir(os.path.join(current_path, subdir))
        if len(dirs) > max_dirs:
            dir_summaries["_more_dirs"] = f"... and {len(dirs) - max_dirs} more"
        # summarize files
        file_summary = {}
        if bulk:
            display = bulk[:max_files]
            if len(bulk) > max_files:
                display.append(f"... and {len(bulk) - max_files} more")
            file_summary["_bulk"] = display
        if meta:
            file_summary["_meta"] = meta
        summary.update(dir_summaries)
        summary.update(file_summary)
        return summary

    return summarize_dir(path)


def print_summary(summary, indent=0):
    for name, content in summary.items():
        if name == "_bulk":
            for f in content:
                print("    " * indent + f"- [bulk] {f}")
        elif name == "_meta":
            for f in content:
                print("    " * indent + f"- [meta] {f}")
        elif name == "_more_dirs":
            print("    " * indent + content)
        elif name == "_error":
            print("    " * indent + f"- [error] {content}")
        else:
            print("    " * indent + f"{name}/")
            print_summary(content, indent + 1)


PATH = "/u/someki1/nvme/espnets/i3d/egs2/europarl/asr1/downloads/v1.1"

tree = summarize_tree(PATH, max_dirs=2, max_files=4)
print_summary(tree)
