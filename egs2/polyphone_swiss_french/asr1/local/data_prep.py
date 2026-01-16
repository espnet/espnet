#!/usr/bin/env python3
import os
import pathlib
import re
import subprocess
import sys
from collections import defaultdict
from random import shuffle


class FrPolyphonePrepper:
    """Data preparation script for the Swiss French Polyphone corpus.

    This class provides the scripting backbone for preparing the
    Swiss French Polyphone corpus for ASR training with espnet.  This
    script was written in reference to the following version of the
    corpus as listed on ELRA:

    This class/script, together with the corresponding configuration
    file `conf/dataprep.yaml` perform all required data prep.

    .. _ELRA-S0030_02:
        http://catalog.elra.info/en-us/repository/browse/ELRA-S0030_02

    """

    def __init__(self, datadir, trainlist=None, devlist=None, testlist=None):
        """Initializes a new instance of the dataprepper class.

        This __init__ function requires a root datadir, and optionally takes
        pre-set train/dev/test lists as arguments for experiment replication.
        If the lists are not provided, new, randomized partitions are created.
        The Swiss French Polyphone corpus provides detailed speaker information,
        thus all splits take into account speaker ID information by default.

        Args:
            datadir (str): The output data directory location.
            trainlist (str): Optional fixed list of training utt IDs.
            devlist (str): Optional fixed list of dev utt IDs.
            testlist (str): Optional fixed list of test utt IDs.

        """
        self.datadir = datadir
        self.valid = 0
        self.audiocorpus = {}
        self.references = {}
        self.all_chars = defaultdict(int)
        self.trainlist = trainlist
        self.devlist = devlist
        self.testlist = testlist
        self._loadlists()

    def _loadlists(self):
        """Helper function to load pre-defined train/dev/test lists.

        Helper function to load pre-defined train/dev/test lists if
        provided by the user at initialization time.

        """
        trainlist = set([])
        with open(self.trainlist) as ifp:
            for fid in ifp:
                trainlist.add(fid.strip())
        self.trainlist = trainlist

        devlist = set([])
        with open(self.devlist) as ifp:
            for fid in ifp:
                devlist.add(fid.strip())
        self.devlist = devlist

        testlist = set([])
        with open(self.testlist) as ifp:
            for fid in ifp:
                testlist.add(fid.strip())
        self.testlist = testlist

        return

    def _cleantext(self, text):
        """Preprocess and cleanup the text as needed.

        Preprocess and cleanup the text as needed.  These regexes perform
        some standard cleanup of the otherwise noisy transcriptions, and
        are also used to remove all `event` markers, which are not used by
        espnet during training.  After running all cleanup, any utterances
        which result in empty transcriptions are subsequently discarded
        and not used for training, development or testing.

        Args:
            text (str): Input transcription candidate for normalization.

        Returns:
            str: The normalized, possibly empty transcription candidate.

        """
        # Skip digits
        if re.match(r"^.*[0-9].*$", text):
            return ""

        text = text.lower()
        text = re.sub(r"\[\\?hésitation", " ", text)
        text = re.sub(r"\[\\?prononciation bizarre", " ", text)
        text = re.sub(r"\[\\?inintelligible", " ", text)
        text = re.sub(r"\[[^\]]+\]", " ", text)
        text = re.sub(r"[\[\]]+", " ", text)
        text = re.sub(r"[\-]+", " ", text)
        text = re.sub(r"’+", "'", text)
        text = re.sub(r"`+", "'", text)
        text = re.sub(r"ҫœ", "oe", text)
        text = re.sub(r"ò", "o", text)
        text = re.sub(r"ҫ", "ç", text)
        text = re.sub(r"[º\"\>«ʿҫœ.\—&š,(đáñ?\_;čó£§žø!ż]+", " ", text)
        text = re.sub(r"[ˢ…^ńı\|ā/“:½\–=*»ßł”°ÿ\}\)í\{ú\$]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        for c in list(text):
            self.all_chars[c] += 1

        return text

    def printcharcounts(self):
        """Utility function for dumping unique character counts.

        Utility function for dumping accumulated unique character counts.
        This is useful for validating and updating the normalization
        routines defined in `_cleantext`

        """
        for key, val in sorted(self.all_chars.items(), key=lambda x: x[1]):
            print(key, val)

    def _processline(self, line):
        """Process a single input line from the corpus.

        Process a single input line from the corpus.  These lines include
        the `uttid` and `text` transcription, while the `uttid` includes
        a `speakerid` and `gender` identifier.  Here we break the pieces
        out and further pass the text to `_cleantext` for normalization.

        Args:
            line (str): The input line including uttid and transcription.

        Returns:
            uttid (str): The uutterance ID.
            text (str): The normalized transcription.
            gender (str): The gender ID f/m.
            spkrid (str): The speaker ID[first 5 characters of the uttid].

        """
        uttid, text = re.split(r"\t", line.strip())
        spkrid = uttid[0:5]
        gender = uttid[0].lower()
        text = self._cleantext(text)

        return uttid, text, gender, spkrid

    def fixdirnames(self, folderroot):
        """Remove spaces, dashes, and commas from folder names.

        The default directory naming conventions for the corpus are not
        particularly friendly to automatic processing.  This function
        simply translates spaces/commas/dashes to a consistent underscore
        pattern for simplicity.

        Args:
            folderroot (str): The folder root to start processing from.

        """
        for folder in os.listdir(folderroot):
            if not folder.startswith("Polyphone") or not os.path.isdir(
                os.path.join(folderroot, folder)
            ):
                continue
            nospace = re.sub(r"\s+", "_", folder)
            nospace = re.sub(r"\-", "_", nospace)
            nospace = re.sub(r",", "_", nospace)
            nospace = re.sub(r"_+", "_", nospace)
            folderpath = os.path.join(folderroot, folder)
            nospacepath = os.path.join(folderroot, nospace)

            if not folderpath == nospacepath:
                os.rename(folderpath, nospace)

        return

    def maprefstofiles(self, ofpath):
        """Map all valid references to corresponding files.

        Map all valid references to corresponding files.

        """
        with open(ofpath, "w", encoding="utf-8") as ofp:
            for fid, fpath in self.audiocorpus.items():
                if fid in self.references:
                    trans = "{0}\t{1}".format(fid, self.references[fid])
                    print(trans, file=ofp)
                    self.valid += 1

        print(
            "Wrote: {0} potential targets to {1}.".format(self.valid, ofpath),
            file=sys.stderr,
        )

        return

    def guess_encoding(self, infile):
        """Guess the encoding type.

        Guess the file encoding type.  Some are DOS850, but most are
        actually just iso-8859-1.  This is actually independent of which
        directory the files are in.  Some DOS are still iso-8859-1.
        """
        mime = subprocess.Popen(
            "/usr/bin/file --mime {0}".format(infile),
            shell=True,
            stdout=subprocess.PIPE,
        ).communicate()[0]

        mime = re.split(r"\s+", mime.decode("utf8").strip())[-1]
        mime = re.sub(r"^charset=", "", mime)

        if mime == "unknown-8bit":
            return "850"

        return mime

    def processreference(self, referencefile):
        """Process a single reference file.

        Process a single reference file.  Each file covers the full
        contribution of a single speaker, which comprises multiple
        recordings and transcriptions.

        Args:
            referencefile(str): The input reference file for mapping.

        """

        encoding = self.guess_encoding(referencefile)

        with open(referencefile, encoding=encoding) as ifp:
            for line in ifp:
                parts = re.split(r"\s+", line.strip())
                fid = parts.pop(0).upper()
                fid = os.path.split(fid)[-1]
                fid = re.sub(r"\.ALW$", "", fid)
                transcription = " ".join(parts)
                self.references[fid] = transcription

        return

    def findfiles(self, path):
        """Recursively search the target dir for reference files.

        Recursively search the root target directory for reference files,
        then process each valid file in turn.  There is some inconsistency
        in the DOS/UNIX sets, though they largely overlap.  We process
        everything but only retain the global unique set[no dupes].

        Args:
            path(str): The root path to start the recursion from.

        """
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                # .ALW are audio files
                if fpath.endswith(".ALW"):
                    fid = os.path.split(fpath)[-1]
                    fid = re.sub(r"\.ALW$", "", fid)
                    fpath = re.sub(r"^\./", "", fpath)
                    self.audiocorpus[fid] = fpath
                # .LST files contain transcriptions
                elif fpath.endswith(".LST"):
                    if "/DOS/" in fpath:
                        self.processreference(fpath)
                    elif "/UNIX/" in fpath:
                        self.processreference(fpath)
                    else:
                        pass
            else:
                self.findfiles(fpath)

    def _collatecorpusdata(self, datadir, corpus):
        """Collate all corpus data.

        Collate all specified corpus data.  This is a hack to call the
        Kaldi collation scripts, which are required by the the other
        prep tools.  It should be simple to remove them once they are
        no longer required.

        Args:
            datadir(str): The target data directory.
            corpus(str): The target corpus(train/dev/test typically).

        """
        command = "utils/fix_data_dir.sh {0}"
        command = command.format(os.path.join(datadir, corpus))
        print(command)
        subprocess.call(
            ["export LC_COLLATE='C';export LC_ALL='C'; {0}".format(command)], shell=True
        )

        return

    def _loadconf(self, fbank_conf):
        """Load the filterbank configuration file.

        Load the filterbank configuration file.  Here we are still
        wantonly specifying 16kHz but we might want to  make an 8kHz
        variant one day.

        Args:
            fbank_conf(str): The filterbank yaml config file.

        Returns:
            dict: The parsed yaml configuration file.  It only check srate.

        """
        conf = {}
        with open(fbank_conf) as ifp:
            for line in ifp:
                flag, val = line.strip().split("=")
                conf[flag] = val

        if not conf.get("--sample-frequency", None):
            conf["--sample-frequency"] = 16000
        else:
            conf["--sample-frequency"] = int(conf["--sample-frequency"])

        return conf

    def processcorpus(self, trans, fbank_conf):
        """The primary corpus processing entrypoint.

        The primary corpus processing entrypoint. There is a lot
        of boilerplate here.  This method organizes the data into
        the basic structures expected by the kaldi(and espnet)
        example training recipes.

        Args:
            trans(str): File containing complete corpus listing.
            fbank_conf(str): The filterbank yaml config file.

        """
        fbank = self._loadconf(fbank_conf)
        datadir = "data"
        text_file = trans
        out_dir = os.path.join(datadir, "preprocess")
        spk2utt = defaultdict(list)
        if not os.path.isdir(out_dir):
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        spkr_ofp = open(os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8")
        gndr_ofp = open(os.path.join(out_dir, "utt2gender"), "w", encoding="utf-8")
        text_ofp = open(os.path.join(out_dir, "text"), "w", encoding="utf-8")
        wav_ofp = open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8")
        with open(text_file) as ifp:
            for line in ifp:
                uttid, text, gender, spkrid = self._processline(line)

                if re.match(r"^\s*$", text):
                    print("Empty utterance: {0}".format(uttid), file=sys.stderr)
                    continue

                spk2utt[spkrid].append(uttid)
                print("{0} {1}".format(uttid, spkrid), file=spkr_ofp)
                print("{0} {1}".format(uttid, gender), file=gndr_ofp)
                print("{0} {1}".format(uttid, text), file=text_ofp)
                sample = "{0} ffmpeg -i {1} -f wav -ar {2} -ab 16 - |"
                sample = sample.format(
                    uttid, self.audiocorpus[uttid], fbank["--sample-frequency"]
                )
                print(sample, file=wav_ofp)

        spkr_ofp.close()
        gndr_ofp.close()
        text_ofp.close()
        wav_ofp.close()

        spkr2utt_ofp = open(os.path.join(out_dir, "spk2utt"), "w")
        for key, vals in spk2utt.items():
            print("{0} {1}".format(key, " ".join(vals)), file=spkr2utt_ofp)
        spkr2utt_ofp.close()

        # This is a necessary evil.
        self._collatecorpusdata("data", "preprocess")

        return

    def make_train_dev_test(self, train, dev, test, splits, useexisting):
        """Create or read the train/dev/test splits.

        Create or read in the existing train/dev/test splits.  If the reference
        lists are provided for replication, then these are used directly.
        Otherwise the lists are generated randomly from the unique list of
        speaker IDs to ensure fully held-out data for dev/test.

        Args:
            train (str): The target path for training data.
            dev (str): The target path for development data.
            test (str): The target path for test data.
            splits (str): A comma separated list with 3 vals that sum to 1.0.
            useexisting (bool): Use existing replication lists or not.

        """
        datadir = self.datadir
        train_path = os.path.join(datadir, train)
        dev_path = os.path.join(datadir, dev)
        test_path = os.path.join(datadir, test)

        for path in [train_path, dev_path, test_path]:
            if not os.path.isdir(path):
                os.mkdir(path)

        corpus = {}
        with open(os.path.join(datadir, "preprocess", "wav.scp")) as ifp:
            for line in ifp:
                parts = re.split(r"\s+", line.strip())
                uttid = parts.pop(0)
                wav = " ".join(parts)
                corpus[uttid] = {"wav": wav}
        with open(os.path.join(datadir, "preprocess", "utt2gender")) as ifp:
            for line in ifp:
                uttid, gender = re.split(r"\s+", line.strip())
                corpus[uttid]["gender"] = gender

        self.spkr2utts = defaultdict(list)
        with open(os.path.join(datadir, "preprocess", "text")) as ifp:
            for line in ifp:
                parts = re.split(r"\s+", line.strip())
                uttid = parts.pop(0)
                text = " ".join(parts)
                corpus[uttid]["text"] = text
                corpus[uttid]["spkrid"] = uttid[0:5]
                self.spkr2utts[uttid[0:5]].append(uttid)

        self.tr_wav_ofp = open(os.path.join(train_path, "wav.scp"), "w")
        self.dv_wav_ofp = open(os.path.join(dev_path, "wav.scp"), "w")
        self.te_wav_ofp = open(os.path.join(test_path, "wav.scp"), "w")

        self.tr_text_ofp = open(os.path.join(train_path, "text"), "w")
        self.dv_text_ofp = open(os.path.join(dev_path, "text"), "w")
        self.te_text_ofp = open(os.path.join(test_path, "text"), "w")

        self.tr_utt2gender_ofp = open(os.path.join(train_path, "utt2gender"), "w")
        self.dv_utt2gender_ofp = open(os.path.join(dev_path, "utt2gender"), "w")
        self.te_utt2gender_ofp = open(os.path.join(test_path, "utt2gender"), "w")

        self.tr_utt2spk_ofp = open(os.path.join(train_path, "utt2spk"), "w")
        self.dv_utt2spk_ofp = open(os.path.join(dev_path, "utt2spk"), "w")
        self.te_utt2spk_ofp = open(os.path.join(test_path, "utt2spk"), "w")

        self.tr_spk2utt_ofp = open(os.path.join(train_path, "spk2utt"), "w")
        self.dv_spk2utt_ofp = open(os.path.join(dev_path, "spk2utt"), "w")
        self.te_spk2utt_ofp = open(os.path.join(test_path, "spk2utt"), "w")

        if useexisting:
            self._replicate_existing(corpus)
        else:
            self._generate_random(corpus, splits)

        self._collatecorpusdata(datadir, train)
        self._collatecorpusdata(datadir, dev)
        self._collatecorpusdata(datadir, test)

        return

    def _replicate_existing(self, corpus):
        """Replicate the original training and eval setup.

        Replicate the original training and eval setup using the
        provided speaker ID lists for train/dev/test.

        Args:
            corpus (:obj:`dict`): The parsed, merged corpus to split.

        """
        items = list(self.spkr2utts.items())

        for idx, spkr_item in enumerate(items):
            for d_item in spkr_item[1]:
                item = [d_item, corpus[d_item]]
                wav_line = "{0} {1}".format(item[0], item[1]["wav"])
                text_line = "{0} {1}".format(item[0], item[1]["text"])
                utt2gender_line = "{0} {1}".format(item[0], item[1]["gender"])
                utt2spk_line = "{0} {1}".format(item[0], spkr_item[0])

                if item[0] in self.trainlist:
                    print(wav_line, file=self.tr_wav_ofp)
                    print(text_line, file=self.tr_text_ofp)
                    print(utt2gender_line, file=self.tr_utt2gender_ofp)
                    print(utt2spk_line, file=self.tr_utt2spk_ofp)
                elif item[0] in self.devlist:
                    print(wav_line, file=self.dv_wav_ofp)
                    print(text_line, file=self.dv_text_ofp)
                    print(utt2gender_line, file=self.dv_utt2gender_ofp)
                    print(utt2spk_line, file=self.dv_utt2spk_ofp)
                elif item[0] in self.testlist:
                    print(wav_line, file=self.te_wav_ofp)
                    print(text_line, file=self.te_text_ofp)
                    print(utt2gender_line, file=self.te_utt2gender_ofp)
                    print(utt2spk_line, file=self.te_utt2spk_ofp)

            spk2utt_line = "{0} {1}".format(spkr_item[0], " ".join(spkr_item[1]))
            if item[0] in self.trainlist:
                print(spk2utt_line, file=self.tr_spk2utt_ofp)
            elif item[0] in self.devlist:
                print(spk2utt_line, file=self.dv_spk2utt_ofp)
            else:
                print(spk2utt_line, file=self.te_spk2utt_ofp)
        return

    def _generate_random(self, corpus, splits):
        """Generate a new, randomized partition.

        Generate a new, randomized partition for the corpus for
        train/dev/test using the breakdowns provided in `splits`.

        Args:
            corpus(:obj:`dict`): The parsed, merged corpus to split.
            splits(str): A comma separated list of train/dev/test sum to 1.0.

        """
        tr, dv, te = [float(val) for val in splits.split(",")]
        count = len(self.spkr2utts.keys())
        tr = int(tr * count)
        dv = int(dv * count)
        te = count - dv - tr
        print(tr, dv, te)

        items = list(self.spkr2utts.items())
        shuffle(items)

        for idx, spkr_item in enumerate(items):
            for d_item in spkr_item[1]:
                item = [d_item, corpus[d_item]]
                wav_line = "{0} {1}".format(item[0], item[1]["wav"])
                text_line = "{0} {1}".format(item[0], item[1]["text"])
                utt2gender_line = "{0} {1}".format(item[0], item[1]["gender"])
                utt2spk_line = "{0} {1}".format(item[0], spkr_item[0])

                if idx >= 0 and idx < tr:
                    print(wav_line, file=self.tr_wav_ofp)
                    print(text_line, file=self.tr_text_ofp)
                    print(utt2gender_line, file=self.tr_utt2gender_ofp)
                    print(utt2spk_line, file=self.tr_utt2spk_ofp)
                elif idx >= tr and idx < tr + dv:
                    print(wav_line, file=self.dv_wav_ofp)
                    print(text_line, file=self.dv_text_ofp)
                    print(utt2gender_line, file=self.dv_utt2gender_ofp)
                    print(utt2spk_line, file=self.dv_utt2spk_ofp)
                else:
                    print(wav_line, file=self.te_wav_ofp)
                    print(text_line, file=self.te_text_ofp)
                    print(utt2gender_line, file=self.te_utt2gender_ofp)
                    print(utt2spk_line, file=self.te_utt2spk_ofp)

            spk2utt_line = "{0} {1}".format(spkr_item[0], " ".join(spkr_item[1]))
            if idx >= 0 and idx < tr:
                print(spk2utt_line, file=self.tr_spk2utt_ofp)
            elif idx >= tr and idx < tr + dv:
                print(spk2utt_line, file=self.dv_spk2utt_ofp)
            else:
                print(spk2utt_line, file=self.te_spk2utt_ofp)

        return


if __name__ == "__main__":
    import argparse

    import yaml

    example = "{0} --config conf/dataprep.yml".format(sys.argv[0])
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument("--config", "-c", help="YAML config file.", required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.Loader)

    mapper = FrPolyphonePrepper(
        datadir=config.get("datadir", "data"),
        trainlist=config.get("trainlist", None),
        devlist=config.get("devlist", None),
        testlist=config.get("testlist", None),
    )
    mapper.fixdirnames(config["download_dir"])
    mapper.findfiles(config["download_dir"])
    mapper.maprefstofiles(config["trans_file"])
    mapper.processcorpus(config["trans_file"], config["fbank_conf"])
    mapper.make_train_dev_test(
        config["train"],
        config["dev"],
        config["test"],
        config["splits"],
        config["useexisting"],
    )
