import os
import sys
import tarfile
import zipfile
from datetime import datetime
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import yaml


class Archiver:
    """
        Archiver class for handling different types of archive files, including .tar,
    .tgz, .tbz2, .txz, and .zip formats. This class provides methods for adding
    files, extracting files, and generating metadata for archived files.

    Attributes:
        type (str): The type of the archive ('tar' or 'zip').
        fopen: The opened archive file object.

    Args:
        file (Union[str, Path]): The path to the archive file.
        mode (str): The mode in which to open the archive. Default is 'r'.

    Raises:
        ValueError: If the archive format cannot be detected or is not supported.

    Examples:
        Creating a new archive:
            >>> with Archiver('example.tar', mode='w') as archiver:
            ...     archiver.add('file1.txt')
            ...     archiver.add('file2.txt')

        Extracting files from an archive:
            >>> with Archiver('example.tar', mode='r') as archiver:
            ...     for file_info in archiver:
            ...         archiver.extract(file_info, path='output_directory')
    """

    def __init__(self, file, mode="r"):
        if Path(file).suffix == ".tar":
            self.type = "tar"
        elif Path(file).suffix == ".tgz" or Path(file).suffixes == [".tar", ".gz"]:
            self.type = "tar"
            if mode == "w":
                mode = "w:gz"
        elif Path(file).suffix == ".tbz2" or Path(file).suffixes == [".tar", ".bz2"]:
            self.type = "tar"
            if mode == "w":
                mode = "w:bz2"
        elif Path(file).suffix == ".txz" or Path(file).suffixes == [".tar", ".xz"]:
            self.type = "tar"
            if mode == "w":
                mode = "w:xz"
        elif Path(file).suffix == ".zip":
            self.type = "zip"
        else:
            raise ValueError(f"Cannot detect archive format: type={file}")

        if self.type == "tar":
            self.fopen = tarfile.open(file, mode=mode)
        elif self.type == "zip":
            self.fopen = zipfile.ZipFile(file, mode=mode)
        else:
            raise ValueError(f"Not supported: type={type}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fopen.close()

    def close(self):
        """
        Close the opened archive file.

        This method ensures that the archive file is properly closed,
        releasing any resources associated with it. It is recommended to
        use this method when you are done working with the archive to
        avoid potential data corruption or resource leaks.

        Examples:
            with Archiver("example.tar", "w") as archive:
                # Add files to the archive
                archive.add("file1.txt")
                archive.add("file2.txt")
                # Automatically closes the archive upon exiting the block
            # If using close explicitly
            archive = Archiver("example.zip", "w")
            archive.add("file1.txt")
            archive.close()  # Ensure to close the archive manually
        """
        self.fopen.close()

    def __iter__(self):
        if self.type == "tar":
            return iter(self.fopen)
        elif self.type == "zip":
            return iter(self.fopen.infolist())
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def add(self, filename, arcname=None, recursive: bool = True):
        """
            Add a file or directory to the archive.

        This method adds the specified file to the archive. If the provided
        filename is a directory and recursive addition is enabled, all files
        within that directory will be added to the archive.

        Args:
            filename (str): The path to the file or directory to be added.
            arcname (Optional[str]): The name to use for the file in the archive.
                If not specified, the original filename will be used.
            recursive (bool): Whether to add files recursively if the filename
                is a directory. Defaults to True.

        Raises:
            ValueError: If the archive type is unsupported.

        Examples:
            To add a single file to the archive:

            >>> archiver.add("example.txt")

            To add a directory and its contents:

            >>> archiver.add("my_directory")

            To specify an arcname:

            >>> archiver.add("example.txt", arcname="new_name.txt")
        """
        if arcname is not None:
            print(f"adding: {arcname}")
        else:
            print(f"adding: {filename}")

        if recursive and Path(filename).is_dir():
            for f in Path(filename).glob("**/*"):
                if f.is_dir():
                    continue

                if arcname is not None:
                    _arcname = Path(arcname) / f
                else:
                    _arcname = None

                self.add(f, _arcname)
            return

        if self.type == "tar":
            return self.fopen.add(filename, arcname)
        elif self.type == "zip":
            return self.fopen.write(filename, arcname)
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def addfile(self, info, fileobj):
        """
            Add a file to the archive with the specified information.

        This method allows you to add a file to the archive using a provided
        `info` object that contains metadata about the file, as well as the
        actual file object to be added. The method handles both tar and zip
        archives, utilizing their respective APIs for adding files.

        Args:
            info (TarInfo or ZipInfo): An object containing metadata about the
                file to be added. For tar archives, this should be a
                `tarfile.TarInfo` instance. For zip archives, it should be a
                `zipfile.ZipInfo` instance.
            fileobj (IO): A file-like object that contains the content to be
                added to the archive. It should support the `read()` method.

        Returns:
            None

        Raises:
            ValueError: If the archive type is not supported.

        Examples:
            >>> with Archiver('example.tar', 'w') as archive:
            ...     info = archive.generate_info('example.txt', 100)
            ...     with open('example.txt', 'rb') as f:
            ...         archive.addfile(info, f)

        Note:
            Ensure that the `info` object is correctly populated with the
            necessary metadata, such as name and size, before calling this
            method.
        """

        print(f"adding: {self.get_name_from_info(info)}")
        if self.type == "tar":
            return self.fopen.addfile(info, fileobj)
        elif self.type == "zip":
            return self.fopen.writestr(info, fileobj.read())
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def generate_info(self, name, size) -> Union[tarfile.TarInfo, zipfile.ZipInfo]:
        """
            Generate TarInfo or ZipInfo for a file using system information.

        This method creates metadata for files being added to an archive,
        including the file name, size, and system user/group IDs for tar
        archives. For zip archives, it captures the file size and timestamp.

        Args:
            name (str): The name of the file to be added to the archive.
            size (int): The size of the file in bytes.

        Returns:
            Union[tarfile.TarInfo, zipfile.ZipInfo]: An instance of
            TarInfo or ZipInfo containing the file metadata.

        Raises:
            ValueError: If the archive type is not supported.

        Examples:
            For a tar archive:

            >>> info = archiver.generate_info("example.txt", 1234)

            For a zip archive:

            >>> info = archiver.generate_info("example.zip", 5678)

        Note:
            The generated TarInfo or ZipInfo will include the current
            timestamp as the modification time and the file size as
            specified.
        """
        if self.type == "tar":
            tarinfo = tarfile.TarInfo(str(name))
            if os.name == "posix":
                tarinfo.gid = os.getgid()
                tarinfo.uid = os.getuid()
            tarinfo.mtime = datetime.now().timestamp()
            tarinfo.size = size
            # Keep mode as default
            return tarinfo
        elif self.type == "zip":
            zipinfo = zipfile.ZipInfo(str(name), datetime.now().timetuple()[:6])
            zipinfo.file_size = size
            return zipinfo
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def get_name_from_info(self, info):
        """
        Retrieve the name of the file or directory from the given info object.

        This method extracts the name of the file or directory represented by the
        `info` parameter, which can be of type `tarfile.TarInfo` or `zipfile.ZipInfo`
        depending on the archive type. It raises a ValueError if the archive type is
        not supported.

        Args:
            info (Union[tarfile.TarInfo, zipfile.ZipInfo]): The info object
                containing the metadata of the file or directory.

        Returns:
            str: The name of the file or directory.

        Raises:
            ValueError: If the archive type is not supported.

        Examples:
            >>> import tarfile
            >>> with tarfile.open('example.tar') as tar:
            ...     info = tar.gettarinfo('file.txt')
            ...     name = get_name_from_info(info)
            ...     print(name)
            file.txt

            >>> import zipfile
            >>> with zipfile.ZipFile('example.zip') as zipf:
            ...     info = zipf.getinfo('file.txt')
            ...     name = get_name_from_info(info)
            ...     print(name)
            file.txt
        """
        if self.type == "tar":
            assert isinstance(info, tarfile.TarInfo), type(info)
            return info.name
        elif self.type == "zip":
            assert isinstance(info, zipfile.ZipInfo), type(info)
            return info.filename
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def extract(self, info, path=None):
        """
            Extract a member from the archive.

        This method is used to extract a file or directory from the archive to a
        specified path. The method automatically handles both tar and zip archive
        formats.

        Args:
            info: The archive member to extract. This can be a `TarInfo` or
                `ZipInfo` object depending on the archive type.
            path: The destination path to which the member should be extracted.
                If not specified, the member will be extracted to the current
                working directory.

        Returns:
            The path to the extracted file or directory.

        Raises:
            ValueError: If the archive type is not supported.

        Examples:
            >>> with Archiver('example.tar') as archive:
            ...     archive.extract('file.txt', path='output_dir')
            'output_dir/file.txt'

            >>> with Archiver('example.zip') as archive:
            ...     archive.extract('file.txt', path='output_dir')
            'output_dir/file.txt'
        """
        if self.type == "tar":
            return self.fopen.extract(info, path)
        elif self.type == "zip":
            return self.fopen.extract(info, path)
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def extractfile(self, info, mode="r"):
        """
            Extract a file from the archive.

        This method retrieves a file from the archive specified by the
        provided `info` argument. The `mode` argument can be used to
        specify how the file is opened.

        Args:
            info: The archive member to extract. This can be a TarInfo
                  object for tar archives or a ZipInfo object for zip
                  archives.
            mode: The mode in which to open the file. This defaults to
                  "r" for text mode. For zip files, "rb" will be
                  automatically converted to "r".

        Returns:
            A file-like object corresponding to the extracted file.
            If the mode is "r", a TextIOWrapper is returned, allowing
            for reading the file as text. Otherwise, a binary file-like
            object is returned.

        Raises:
            ValueError: If the archive type is not supported.

        Examples:
            >>> with Archiver("example.zip") as archive:
            ...     with archive.extractfile("file.txt") as f:
            ...         content = f.read()
            ...         print(content)

            >>> with Archiver("example.tar") as archive:
            ...     with archive.extractfile(archive.get_name_from_info(info)) as f:
            ...         data = f.read()
        """
        if self.type == "tar":
            f = self.fopen.extractfile(info)
            if mode == "r":
                return TextIOWrapper(f)
            else:
                return f
        elif self.type == "zip":
            if mode == "rb":
                mode = "r"
            return self.fopen.open(info, mode)
        else:
            raise ValueError(f"Not supported: type={self.type}")


def find_path_and_change_it_recursive(value, src: str, tgt: str):
    """
        Recursively find and replace a specified source path with a target path in a
    given data structure.

    This function can handle nested dictionaries and lists, changing all instances
    of the source path to the target path if they are found.

    Args:
        value (Union[dict, list, tuple, str]): The data structure to search
            through, which can be a dictionary, list, tuple, or string.
        src (str): The source path to search for.
        tgt (str): The target path to replace the source path with.

    Returns:
        Union[dict, list, tuple, str]: The modified data structure with all
            occurrences of the source path replaced by the target path.

    Examples:
        >>> find_path_and_change_it_recursive({'path': '/old/path'}, '/old/path',
        ...                                      '/new/path')
        {'path': '/new/path'}

        >>> find_path_and_change_it_recursive(['/old/path', '/another/path'],
        ...                                      '/old/path', '/new/path')
        ['/new/path', '/another/path']

        >>> find_path_and_change_it_recursive('This is a string with /old/path',
        ...                                      '/old/path', '/new/path')
        'This is a string with /new/path'
    """
    if isinstance(value, dict):
        return {
            k: find_path_and_change_it_recursive(v, src, tgt) for k, v in value.items()
        }
    elif isinstance(value, (list, tuple)):
        return [find_path_and_change_it_recursive(v, src, tgt) for v in value]
    elif isinstance(value, str) and Path(value) == Path(src):
        return tgt
    else:
        return value


def get_dict_from_cache(meta: Union[Path, str]) -> Optional[Dict[str, str]]:
    """
        Retrieve a dictionary from a YAML cache file.

    This function reads a YAML file located at the specified path and returns a
    dictionary mapping keys to file paths. The function ensures that the paths
    exist in the expected output directory.

    Args:
        meta (Union[Path, str]): The path to the YAML cache file.

    Returns:
        Optional[Dict[str, str]]: A dictionary containing key-value pairs from the
        YAML file, where keys are identifiers and values are file paths. Returns
        None if the YAML file does not exist or if any referenced file does not
        exist in the output directory.

    Raises:
        AssertionError: If the loaded YAML content is not a dictionary or if
        the expected keys are not found.

    Examples:
        >>> cache_dict = get_dict_from_cache("path/to/meta.yaml")
        >>> print(cache_dict)
        {'asr_model_file': 'output/model.pth', 'other_file': 'output/other.file'}

    Note:
        This function assumes the structure of the YAML file contains two
        dictionaries under the keys "yaml_files" and "files".
    """
    meta = Path(meta)
    outpath = meta.parent.parent
    if not meta.exists():
        return None

    with meta.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
        assert isinstance(d, dict), type(d)
        yaml_files = d["yaml_files"]
        files = d["files"]
        assert isinstance(yaml_files, dict), type(yaml_files)
        assert isinstance(files, dict), type(files)

        retval = {}
        for key, value in list(yaml_files.items()) + list(files.items()):
            if not (outpath / value).exists():
                return None
            retval[key] = str(outpath / value)
        return retval


def unpack(
    input_archive: Union[Path, str],
    outpath: Union[Path, str],
    use_cache: bool = True,
) -> Dict[str, str]:
    """
    Scan all files in the archive file and return as a dict of files.

    This function extracts the contents of the specified archive and returns
    a dictionary mapping keys from the archive's metadata to the paths of
    the extracted files. If a `meta.yaml` file is present in the archive,
    its contents will be used to determine the output file paths.

    Args:
        input_archive (Union[Path, str]): The path to the input archive file
            (e.g., .tar, .zip).
        outpath (Union[Path, str]): The output directory where files will be
            extracted.
        use_cache (bool): Whether to use cached output if available. Defaults
            to True.

    Returns:
        Dict[str, str]: A dictionary mapping keys to the paths of the
        extracted files.

    Raises:
        RuntimeError: If the format of the archive is incorrect or if
            `meta.yaml` is not found.

    Examples:
        tarfile:
           model.pth
           some1.file
           some2.file

        >>> unpack("tarfile", "out")
        {'asr_model_file': 'out/model.pth'}
    """
    input_archive = Path(input_archive)
    outpath = Path(outpath)

    with Archiver(input_archive) as archive:
        for info in archive:
            if Path(archive.get_name_from_info(info)).name == "meta.yaml":
                if (
                    use_cache
                    and (outpath / Path(archive.get_name_from_info(info))).exists()
                ):
                    retval = get_dict_from_cache(
                        outpath / Path(archive.get_name_from_info(info))
                    )
                    if retval is not None:
                        return retval
                d = yaml.safe_load(archive.extractfile(info))
                assert isinstance(d, dict), type(d)
                yaml_files = d["yaml_files"]
                files = d["files"]
                assert isinstance(yaml_files, dict), type(yaml_files)
                assert isinstance(files, dict), type(files)
                break
        else:
            raise RuntimeError("Format error: not found meta.yaml")

        for info in archive:
            fname = archive.get_name_from_info(info)
            outname = outpath / fname
            outname.parent.mkdir(parents=True, exist_ok=True)
            if fname in set(yaml_files.values()):
                d = yaml.safe_load(archive.extractfile(info))
                # Rewrite yaml
                for info2 in archive:
                    name = archive.get_name_from_info(info2)
                    d = find_path_and_change_it_recursive(d, name, str(outpath / name))
                with outname.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(d, f)
            else:
                archive.extract(info, path=outpath)

        retval = {}
        for key, value in list(yaml_files.items()) + list(files.items()):
            retval[key] = str(outpath / value)
        return retval


def _to_relative_or_resolve(f):
    # Resolve to avoid symbolic link
    p = Path(f).resolve()
    try:
        # Change to relative if it can
        p = p.relative_to(Path(".").resolve())
    except ValueError:
        pass
    return str(p)


def pack(
    files: Dict[str, Union[str, Path]],
    yaml_files: Dict[str, Union[str, Path]],
    outpath: Union[str, Path],
    option: Iterable[Union[str, Path]] = (),
):
    """
        Pack files and YAML files into an archive, creating a metadata YAML file.

    This function validates the existence of the specified files and directories,
    transforms their paths to relative or resolved paths, and then creates an
    archive containing all files along with a metadata file that includes details
    such as the current timestamp and the versions of Python and any installed
    packages.

    Args:
        files (Dict[str, Union[str, Path]]): A dictionary mapping keys to file
            paths that need to be packed.
        yaml_files (Dict[str, Union[str, Path]]): A dictionary mapping keys to
            YAML file paths that need to be packed.
        outpath (Union[str, Path]): The output path for the created archive.
        option (Iterable[Union[str, Path]], optional): Additional files or
            directories to include in the archive. Defaults to an empty tuple.

    Raises:
        FileNotFoundError: If any of the specified files or directories do not
            exist.

    Examples:
        >>> pack(
        ...     files={'model': 'model.pth'},
        ...     yaml_files={'config': 'config.yaml'},
        ...     outpath='output.tar',
        ...     option=['extra_file.txt']
        ... )
        Generate: output.tar
    """
    for v in list(files.values()) + list(yaml_files.values()) + list(option):
        if not Path(v).exists():
            raise FileNotFoundError(f"No such file or directory: {v}")

    files = {k: _to_relative_or_resolve(v) for k, v in files.items()}
    yaml_files = {k: _to_relative_or_resolve(v) for k, v in yaml_files.items()}
    option = [_to_relative_or_resolve(v) for v in option]

    meta_objs = dict(
        files=files,
        yaml_files=yaml_files,
        timestamp=datetime.now().timestamp(),
        python=sys.version,
    )

    try:
        import torch

        meta_objs.update(torch=str(torch.__version__))
    except ImportError:
        pass
    try:
        import espnet

        meta_objs.update(espnet=espnet.__version__)
    except ImportError:
        pass

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with Archiver(outpath, mode="w") as archive:
        # Write packed/meta.yaml
        fileobj = BytesIO(yaml.safe_dump(meta_objs).encode())
        info = archive.generate_info("meta.yaml", fileobj.getbuffer().nbytes)
        archive.addfile(info, fileobj=fileobj)

        for f in list(yaml_files.values()) + list(files.values()) + list(option):
            archive.add(f)

    print(f"Generate: {outpath}")
