import collections.abc
import json
from pathlib import Path
from typing import Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text

try:
    import music21 as m21  # for CI import
except (ImportError, ModuleNotFoundError):
    m21 = None
try:
    import miditoolkit  # for CI import
except (ImportError, ModuleNotFoundError):
    miditoolkit = None


class NOTE(object):
    """
    This class to represent musical notes with associated metadata.

    Attributes:
        lyric (str): The lyric associated with the note.
        midi (int): The MIDI pitch value of the note.
        st (float): The start time of the note in seconds.
        et (float): The end time of the note in seconds.

    Examples:
        >>> note = NOTE("hello", 60, 0.0, 0.5)
        >>> print(note.lyric)
        hello
        >>> print(note.midi)
        60
        >>> print(note.st)
        0.0
        >>> print(note.et)
        0.5
    """

    def __init__(self, lyric, midi, st, et):
        self.lyric = lyric
        self.midi = midi
        self.st = st
        self.et = et


class XMLReader(collections.abc.Mapping):
    """
    Reader class for 'xml.scp'.

    This class allows reading XML files corresponding to musical scores.
    Each key in the 'xml.scp' file maps to a specific XML file, which can be
    parsed to extract musical information such as tempo and notes.

    Attributes:
        fname (Union[Path, str]): The file name or path to the 'xml.scp' file.
        dtype (type): The data type for note representation, default is np.int16.
        data (dict): A dictionary containing the mapping of keys to XML file paths.

    Args:
        fname (Union[Path, str]): Path to the 'xml.scp' file.
        dtype (type): Data type for note representation (default: np.int16).

    Returns:
        None

    Examples:
        Given the following entries in 'xml.scp':
            key1 /some/path/a.xml
            key2 /some/path/b.xml
            key3 /some/path/c.xml
            key4 /some/path/d.xml

        You can access the tempo and note list as follows:

        >>> reader = XMLReader('xml.scp')
        >>> tempo, note_list = reader['key1']

    Raises:
        AssertionError: If the music21 package is not available.

    Note:
        Ensure that the music21 library is installed to use this class.
        If it's not installed, follow the instructions to install the Muskit
        modules via (cd tools && make muskit.done).
    """

    @typechecked
    def __init__(
        self,
        fname: Union[Path, str],
        dtype: type = np.int16,
    ):
        assert m21 is not None, (
            "Cannot load music21 package. ",
            "Please install Muskit modules via ",
            "(cd tools && make muskit.done)",
        )
        self.fname = fname
        self.dtype = dtype
        self.data = read_2columns_text(fname)  # get key-value dict

    def __getitem__(self, key):
        score = m21.converter.parse(self.data[key])
        m = score.metronomeMarkBoundaries()

        # NOTE(Yuxun): tempo with '(playback only)' returns None in item[2].number
        tempo = None
        for item in m:
            tempo_number = item[2].number
            if tempo_number is not None:
                tempo = tempo_number
                break
        tempo = int(tempo)

        part = score.parts[0].flat
        notes_list = []
        prepitch = -1
        st = 0
        for note in part.notesAndRests:
            dur = note.seconds
            if not note.isRest:  # Note or Chord
                lr = note.lyric
                if note.isChord:
                    for n in note:
                        if n.pitch.midi != prepitch:  # Ignore repeat note
                            note = n
                            break
                if lr is None or lr == "" or lr == "ー":  # multi note in one syllable
                    if note.pitch.midi == prepitch or prepitch == 0:  # same pitch
                        notes_list[-1].et += dur
                    else:  # different pitch
                        notes_list.append(NOTE("—", note.pitch.midi, st, st + dur))
                elif lr == "br":  # <br> is tagged as a note
                    if prepitch == 0:
                        notes_list[-1].et += dur
                    else:
                        notes_list.append(NOTE("P", 0, st, st + dur))
                    prepitch = 0
                    st += dur
                    continue
                else:  # normal note for one syllable
                    notes_list.append(NOTE(lr, note.pitch.midi, st, st + dur))
                prepitch = note.pitch.midi
                for arti in note.articulations:
                    # NOTE(Yuning): By default, 'breath mark' appears at the end of
                    # the sentence. In some situations, 'breath mark' doesn't take
                    # effect in its belonging note. Please handle them under local/.
                    if arti.name in ["breath mark"]:  # <br> is tagged as a notation
                        notes_list.append(NOTE("B", 0, st + dur, st + dur))
                    # NOTE(Yuning): In some datasets, there is a break when 'staccato'
                    # occurs. We let users to decide whether to perform segmentation
                    # under local/.
            else:  # rest note
                if prepitch == 0:
                    notes_list[-1].et += dur
                else:
                    notes_list.append(NOTE("P", 0, st, st + dur))
                prepitch = 0
            st += dur
        # NOTE(Yuning): implicit rest at the end of xml file should be removed.
        if notes_list[-1].midi == 0 and notes_list[-1].lyric == "P":
            notes_list.pop()
        return tempo, notes_list

    def get_path(self, key):
        """
            Retrieve the file path associated with a given key.

        This method accesses the internal data dictionary to return the
        path of the XML or MIDI file associated with the provided key.

        Args:
            key (str): The key for which the file path is to be retrieved.

        Returns:
            str: The file path corresponding to the provided key.

        Raises:
            KeyError: If the key is not found in the data dictionary.

        Examples:
            >>> reader = XMLReader('xml.scp')
            >>> path = reader.get_path('key1')
            >>> print(path)
            /some/path/a.xml

        Note:
            Ensure that the key exists in the data before calling this method
            to avoid a KeyError.
        """
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        """
            Reader class for 'xml.scp'.

        This class reads XML files specified in a key-value format from an input
        file (xml.scp). Each key corresponds to a path of an XML file, and the
        class provides methods to access the tempo and a list of notes for each
        key.

        Attributes:
            fname (Union[Path, str]): The path to the input file containing keys
                and their corresponding XML file paths.
            dtype (type): The data type for notes, default is numpy.int16.
            data (dict): A dictionary containing the mapping of keys to XML file
                paths.

        Args:
            fname (Union[Path, str]): The path to the 'xml.scp' file.
            dtype (type, optional): The data type for notes. Defaults to np.int16.

        Returns:
            Tuple[int, List[NOTE]]: The tempo as an integer and a list of NOTE
            objects.

        Raises:
            AssertionError: If the music21 package cannot be imported.

        Examples:
            key1 /some/path/a.xml
            key2 /some/path/b.xml
            key3 /some/path/c.xml
            key4 /some/path/d.xml
            ...

            >>> reader = XMLReader('xml.scp')
            >>> tempo, note_list = reader['key1']
        """
        return self.data.keys()


class XMLWriter:
    """
    Writer class for 'midi.scp'.

    This class is responsible for writing XML representations of musical
    scores to specified files and maintaining a corresponding SCP (score
    control protocol) file that maps keys to file paths.

    Attributes:
        dir (Path): The output directory where XML files will be stored.
        fscp (TextIOWrapper): The file object for writing the SCP file.
        data (dict): A dictionary that maps keys to their corresponding XML
            file paths.

    Args:
        outdir (Union[Path, str]): The directory where XML files will be
            saved.
        scpfile (Union[Path, str]): The path of the SCP file to be written.

    Examples:
        key1 /some/path/a.musicxml
        key2 /some/path/b.musicxml
        key3 /some/path/c.musicxml
        key4 /some/path/d.musicxml
        ...

        >>> writer = XMLWriter('./data/', './data/xml.scp')
        >>> writer['aa'] = xml_obj
        >>> writer['bb'] = xml_obj

    Raises:
        AssertionError: If the value provided does not contain exactly four
            elements (lyrics, notes, segmentations, and tempo) when using
            the __setitem__ method.

    Note:
        Ensure that the music21 library is installed, as it is required for
        creating and writing XML files.
    """

    @typechecked
    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
    ):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.data = {}

    def __setitem__(self, key: str, value: tuple):
        assert (
            len(value) == 4
        ), "The xml values should include lyrics, note, segmentations and tempo"
        lyrics_seq, notes_seq, segs_seq, tempo = value
        xml_path = self.dir / f"{key}.musicxml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)

        m = m21.stream.Stream()
        m.insert(m21.tempo.MetronomeMark(number=tempo))
        bps = 1.0 * tempo / 60
        offset = 0
        for i in range(len(lyrics_seq)):
            duration = int(8 * (segs_seq[i][1] - segs_seq[i][0]) * bps + 0.5)
            duration = 1.0 * duration / 8
            if duration == 0:
                duration = 1 / 16
            if notes_seq[i] != 0:  # isNote
                n = m21.note.Note(notes_seq[i])
                if lyrics_seq[i] != "—":
                    n.lyric = lyrics_seq[i]
            else:  # isRest
                n = m21.note.Rest()
            n.offset = offset
            n.duration = m21.duration.Duration(duration)
            m.insert(n)
            offset += duration
        m.write("xml", fp=xml_path)
        self.fscp.write(f"{key} {xml_path}\n")
        self.data[key] = str(xml_path)

    def get_path(self, key):
        """
        Retrieve the file path associated with a given key.

        This method returns the path to the XML file that corresponds to
        the specified key in the internal data dictionary.

        Args:
            key (str): The key whose associated file path is to be retrieved.

        Returns:
            str: The file path associated with the given key.

        Raises:
            KeyError: If the key does not exist in the data dictionary.

        Examples:
            >>> writer = XMLWriter('./data/', './data/xml.scp')
            >>> writer['example'] = xml_obj
            >>> path = writer.get_path('example')
            >>> print(path)  # Output: ./data/example.musicxml

        Note:
            Ensure that the key exists in the data dictionary before calling
            this method to avoid a KeyError.
        """
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Writer class for 'midi.scp'.

            This class allows for writing MusicXML files and generating a corresponding
            SCP (script) file that maps keys to their respective MusicXML file paths.
            The data is organized in a way that facilitates easy access and storage.

            Attributes:
                dir (Path): The directory where MusicXML files will be stored.
                fscp (TextIOWrapper): The file handle for the SCP file.
                data (dict): A dictionary mapping keys to their corresponding MusicXML paths.

            Args:
                outdir (Union[Path, str]): The output directory for MusicXML files.
                scpfile (Union[Path, str]): The path for the SCP file to be created.

            Examples:
                key1 /some/path/a.musicxml
                key2 /some/path/b.musicxml
                key3 /some/path/c.musicxml
                key4 /some/path/d.musicxml
                ...

                >>> writer = XMLWriter('./data/', './data/xml.scp')
                >>> writer['aa'] = xml_obj
                >>> writer['bb'] = xml_obj

            Raises:
                AssertionError: If the provided value tuple does not contain exactly four
                elements when using `__setitem__`.

            Note:
                The XML values provided to `__setitem__` should include lyrics, notes,
                segmentations, and tempo, in that order.
        """
        self.fscp.close()


class MIDReader(collections.abc.Mapping):
    """
    Reader class for 'mid.scp'.

    This class reads MIDI files and extracts the tempo and note sequences
    from them. The data is organized in a key-value format, where each key
    corresponds to a path of a MIDI file.

    Attributes:
        fname (Union[Path, str]): The filename or path of the 'mid.scp' file.
        dtype (type): The data type to be used for the MIDI values (default: np.int16).
        add_rest (bool): A flag to indicate whether to add rest notes to the
            note sequence (default: True).
        data (dict): A dictionary containing the key-value pairs from the
            'mid.scp' file.

    Args:
        fname (Union[Path, str]): The path to the 'mid.scp' file.
        add_rest (bool, optional): Whether to include rest notes in the
            output. Defaults to True.
        dtype (type, optional): The data type for MIDI values. Defaults to np.int16.

    Returns:
        tuple: A tuple containing the tempo (int) and a list of NOTE objects
            representing the notes in the MIDI file.

    Examples:
        key1 /some/path/a.mid
        key2 /some/path/b.mid
        key3 /some/path/c.mid
        key4 /some/path/d.mid
        ...

        >>> reader = MIDReader('mid.scp')
        >>> tempo, note_list = reader['key1']

    Raises:
        AssertionError: If the miditoolkit package is not available or if the
            tempo changes are not correctly identified in the MIDI file.

    Note:
        The MIDI files do not contain explicit rest notes; however, the
        `add_rest` option allows the insertion of implicit rest notes based on
        the timing of the notes.
    """

    @typechecked
    def __init__(
        self,
        fname: Union[Path, str],
        add_rest: bool = True,
        dtype: type = np.int16,
    ):
        assert miditoolkit is not None, (
            "Cannot load miditoolkit package. ",
            "Please install Muskit modules via ",
            "(cd tools && make muskit.done)",
        )
        self.fname = fname
        self.dtype = dtype
        self.add_rest = add_rest  # add rest into note sequencee
        self.data = read_2columns_text(fname)  # get key-value dict

    def __getitem__(self, key):
        midi_obj = miditoolkit.midi.parser.MidiFile(self.data[key])
        # load tempo
        tempos = midi_obj.tempo_changes
        tempos.sort(key=lambda x: (x.time, x.tempo))
        assert len(tempos) == 1
        tempo = int(tempos[0].tempo + 0.5)
        # load pitch time sequence
        tick_to_time = midi_obj.get_tick_to_time_mapping()
        notes = midi_obj.instruments[0].notes
        notes.sort(key=lambda x: (x.start, x.pitch))
        notes_list = []
        pre_et = 0
        for note in notes:
            st = tick_to_time[note.start]
            et = tick_to_time[note.end]
            # NOTE(Yuning): MIDIs don't have explicit rest notes.
            # Explicit rest notes might be needed for stage 1 in svs.
            if st != pre_et and self.add_rest:
                notes_list.append(NOTE("P", 0, pre_et, st))
            notes_list.append(NOTE("*", note.pitch, st, et))
            pre_et = et
        return tempo, notes_list

    def get_path(self, key):
        """
            Retrieve the file path associated with the given key.

        This method returns the file path for the specified key in the
        mapping. The key must exist in the internal data structure.

        Args:
            key (str): The key for which to retrieve the file path.

        Returns:
            str: The file path associated with the provided key.

        Raises:
            KeyError: If the key is not found in the data.

        Examples:
            >>> reader = MIDReader('mid.scp')
            >>> path = reader.get_path('key1')
            >>> print(path)  # Output: /some/path/a.mid

        Note:
            Ensure that the key exists in the mapping before calling
            this method to avoid a KeyError.
        """
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        """
            Reader class for 'mid.scp'.

        This class provides functionality to read MIDI files and extract
        relevant musical information, including tempo and note sequences.
        It can read a mapping of keys to MIDI file paths from a specified
        file, allowing easy access to the MIDI data.

        Attributes:
            fname (Union[Path, str]): The path to the 'mid.scp' file.
            dtype (type): The data type for note representation (default: np.int16).
            add_rest (bool): Indicates whether to add rest notes into the note
                sequence (default: True).
            data (dict): A dictionary mapping keys to MIDI file paths.

        Args:
            fname (Union[Path, str]): The path to the 'mid.scp' file.
            add_rest (bool, optional): Whether to include rest notes in the output.
                Defaults to True.
            dtype (type, optional): The data type for note representation.
                Defaults to np.int16.

        Returns:
            tempo (int): The tempo of the MIDI file.
            notes_list (list): A list of NOTE objects representing the notes.

        Examples:
            >>> reader = MIDReader('mid.scp')
            >>> tempo, note_list = reader['key1']
            >>> print(tempo)
            120
            >>> print(note_list)
            [<__main__.NOTE object at 0x...>, <__main__.NOTE object at 0x...>, ...]

        Raises:
            AssertionError: If the miditoolkit package is not installed or if
                the MIDI file cannot be parsed.

        Note:
            The MIDReader class uses the miditoolkit package to parse MIDI files.
            Ensure that the package is installed for proper functionality.

        Todo:
            - Implement support for additional MIDI file formats.
            - Add more robust error handling for MIDI parsing.
        """
        return self.data.keys()


class SingingScoreReader(collections.abc.Mapping):
    """
    Reader class for 'score.scp'.

    This class provides an interface to read singing scores from a
    specified file. The file should contain paths to JSON files that
    represent the musical scores.

    Attributes:
        fname (Union[Path, str]): The file name or path of the 'score.scp' file.
        dtype (type): The data type for numerical values (default: np.int16).
        data (dict): A dictionary mapping keys to file paths read from 'score.scp'.

    Args:
        fname (Union[Path, str]): The path to the 'score.scp' file.
        dtype (type, optional): The data type for numerical values (default: np.int16).

    Returns:
        dict: The content of the JSON file corresponding to the key.

    Examples:
        key1 /some/path/score.json
        key2 /some/path/score.json
        key3 /some/path/score.json
        key4 /some/path/score.json
        ...

        >>> reader = SingingScoreReader('score.scp')
        >>> score = reader['key1']

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        KeyError: If the specified key is not found in the data.

    Note:
        Ensure that the 'score.scp' file is properly formatted with valid paths
        to the JSON files.
    """

    @typechecked
    def __init__(
        self,
        fname: Union[Path, str],
        dtype: type = np.int16,
    ):
        self.fname = fname
        self.dtype = dtype
        self.data = read_2columns_text(fname)

    def __getitem__(self, key):
        with open(self.data[key], "r") as f:
            score = json.load(f)
        return score

    def get_path(self, key):
        """
        Retrieve the file path associated with the given key.

        This method returns the path of the file stored in the internal
        data dictionary for the specified key.

        Args:
            key (str): The key whose associated file path is to be retrieved.

        Returns:
            str: The file path associated with the given key.

        Examples:
            >>> reader = XMLReader('xml.scp')
            >>> path = reader.get_path('key1')
            >>> print(path)
            /some/path/a.xml

        Raises:
            KeyError: If the key is not found in the data dictionary.
        """
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        """
            Reader class for 'score.scp'.

        This class allows for reading singing scores stored in JSON format.
        Each score contains a tempo and a list of notes, where each note has
        attributes such as start time, end time, lyric, MIDI pitch, and phoneme.

        Attributes:
            fname (Union[Path, str]): The path to the score file.
            dtype (type): The data type for the score values, default is np.int16.
            data (dict): A dictionary mapping keys to file paths.

        Args:
            fname (Union[Path, str]): The path to the score file.
            dtype (type): The data type for the score values, default is np.int16.

        Returns:
            dict: A dictionary representation of the singing score.

        Examples:
            key1 /some/path/score.json
            key2 /some/path/score.json
            key3 /some/path/score.json
            key4 /some/path/score.json
            ...

            >>> reader = SingingScoreReader('score.scp')
            >>> score = reader['key1']

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If the key is not found in the score file.

        Note:
            Ensure that the score JSON files are structured correctly to avoid
            loading errors.
        """
        return self.data.keys()


class SingingScoreWriter:
    """
        Writer class for 'score.scp'.

    This class allows for writing musical scores to a specified output directory
    and maintaining a corresponding SCP file that maps keys to score file paths.
    The scores are saved in JSON format.

    Attributes:
        dir (Path): The directory where score files will be written.
        fscp (TextIOWrapper): The file handle for the SCP file.
        data (dict): A dictionary to store the mapping of keys to score file paths.

    Args:
        outdir (Union[Path, str]): The output directory for the score files.
        scpfile (Union[Path, str]): The path to the SCP file to be created.

    Examples:
        key1 /some/path/score.json
        key2 /some/path/score.json
        key3 /some/path/score.json
        key4 /some/path/score.json
        ...

        >>> writer = SingingScoreWriter('./data/', './data/score.scp')
        >>> writer['aa'] = score_obj
        >>> writer['bb'] = score_obj

    Methods:
        __setitem__(key: str, value: dict): Saves a score under the specified key.
        get_path(key): Returns the file path associated with the given key.
        __enter__(): Supports the use of the context manager.
        __exit__(exc_type, exc_val, exc_tb): Cleans up by closing the SCP file.
        close(): Closes the SCP file.

    Note:
        The score should be a dictionary with the following structure:

        {
            "tempo": bpm,
            "item_list": a subset of ["st", "et", "lyric", "midi", "phn"],
            "note": [
                [start_time1, end_time1, lyric1, midi1, phn1],
                [start_time2, end_time2, lyric2, midi2, phn2],
                ...
            ]
        }

        The items in each note correspond to the "item_list".
    """

    @typechecked
    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
    ):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.data = {}

    def __setitem__(self, key: str, value: dict):
        """Score should be a dict

        Example:
        {
            "tempo": bpm,
            "item_list": a subset of ["st", "et", "lyric", "midi", "phn"],
            "note": [
                [start_time1, end_time1, lyric1, midi1, phn1],
                [start_time2, end_time2, lyric2, midi2, phn2],
                ...
            ]
        }

        The itmes in each note correspond to the "item_list".

        """

        score_path = self.dir / f"{key}.json"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        self.fscp.write(f"{key} {score_path}\n")
        self.data[key] = str(score_path)

    def get_path(self, key):
        """
            Writer class for 'score.scp'.

        This class is responsible for writing score data to a specified directory
        and maintaining a mapping between keys and their corresponding file paths.

        Attributes:
            dir (Path): The output directory for the score files.
            fscp (TextIOWrapper): The file handle for the SCP file.
            data (dict): A dictionary mapping keys to their respective file paths.

        Args:
            outdir (Union[Path, str]): The output directory where score files will
                be stored.
            scpfile (Union[Path, str]): The path to the SCP file that will be
                created or written to.

        Examples:
            key1 /some/path/score.json
            key2 /some/path/score.json
            key3 /some/path/score.json
            key4 /some/path/score.json
            ...

            >>> writer = SingingScoreWriter('./data/', './data/score.scp')
            >>> writer['aa'] = score_obj
            >>> writer['bb'] = score_obj

        Methods:
            __setitem__(key: str, value: dict):
                Writes the score data as a JSON file and updates the SCP mapping.
            get_path(key):
                Returns the file path associated with the given key.
            __enter__():
                Enables the use of the class in a context manager.
            __exit__(exc_type, exc_val, exc_tb):
                Closes the file handle upon exiting the context manager.
            close():
                Closes the SCP file handle.

        Note:
            The score data should be structured as follows:
            {
                "tempo": bpm,
                "item_list": a subset of ["st", "et", "lyric", "midi", "phn"],
                "note": [
                    [start_time1, end_time1, lyric1, midi1, phn1],
                    [start_time2, end_time2, lyric2, midi2, phn2],
                    ...
                ]
            }
            The items in each note correspond to the "item_list".

        Raises:
            OSError: If the output directory cannot be created or accessed.
        """
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Writer class for 'score.scp'.

            This class allows writing singing score data to a specified directory
            and maintaining a corresponding 'scp' file for easy access to the
            written scores.

            Attributes:
                dir (Path): The directory where score files will be saved.
                fscp (TextIOWrapper): The file object for the 'scp' file.
                data (dict): A dictionary to store paths of written scores.

            Args:
                outdir (Union[Path, str]): The output directory for score files.
                scpfile (Union[Path, str]): The path to the 'scp' file.

            Examples:
                key1 /some/path/score.json
                key2 /some/path/score.json
                key3 /some/path/score.json
                key4 /some/path/score.json
                ...

                >>> writer = SingingScoreWriter('./data/', './data/score.scp')
                >>> writer['aa'] = score_obj
                >>> writer['bb'] = score_obj

            Methods:
                __setitem__(key: str, value: dict):
                    Writes the score data to a JSON file and updates the 'scp' file.

                get_path(key):
                    Returns the path of the score file associated with the given key.

                __enter__():
                    Allows the use of the context manager.

                __exit__(exc_type, exc_val, exc_tb):
                    Closes the file when exiting the context manager.

                close():
                    Closes the 'scp' file.

            Note:
                The score should be a dictionary structured as follows:
                {
                    "tempo": bpm,
                    "item_list": a subset of ["st", "et", "lyric", "midi", "phn"],
                    "note": [
                        [start_time1, end_time1, lyric1, midi1, phn1],
                        [start_time2, end_time2, lyric2, midi2, phn2],
                        ...
                    ]
                }

                The items in each note correspond to the "item_list".
        """
        self.fscp.close()
