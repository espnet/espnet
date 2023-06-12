import os

from .utils import TemporaryDirectory


def validate_converted_files_match(src_file_or_folder, subject, reference_ext=""):
    from .api import extract

    with TemporaryDirectory() as tmpdir:
        if os.path.isdir(src_file_or_folder):
            src_folder = src_file_or_folder
        else:
            extract(
                src_file_or_folder + reference_ext, dest_dir=os.path.join(tmpdir, "src")
            )
            src_folder = os.path.join(tmpdir, "src")

        converted_folder = os.path.join(tmpdir, "converted")
        extract(subject, dest_dir=converted_folder)

        missing_files = set()
        mismatch_size = set()
        for root, dirs, files in os.walk(src_folder):
            for f in files:
                absfile = os.path.join(root, f)
                rp = os.path.relpath(absfile, src_folder)
                destpath = os.path.join(converted_folder, rp)
                if not os.path.islink(destpath):
                    if not os.path.isfile(destpath):
                        missing_files.add(rp)
                    elif os.stat(absfile).st_size != os.stat(destpath).st_size:
                        mismatch_size.add(rp)
    return src_file_or_folder, missing_files, mismatch_size
