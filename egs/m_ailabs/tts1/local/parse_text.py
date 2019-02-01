#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_tag", type=str, default=None, nargs="?",
                        help="language tag (can be used for multi lingual case)")
    parser.add_argument("--spk_tag", type=str,
                        help="speaker tag")
    parser.add_argument("jsons", nargs="+", type=str,
                        help="*_mls.json filenames")
    parser.add_argument("out", type=str,
                        help="output filename")
    args = parser.parse_args()

    dirname = os.path.dirname(args.out)
    if len(dirname) != 0 and not os.path.exists(dirname):
        os.makedirs(dirname)

    with codecs.open(args.out, "w", encoding="utf-8") as out:
        for filename in sorted(args.jsons):
            with codecs.open(filename, "r", encoding="utf-8") as f:
                js = json.load(f)
            for key in sorted(js.keys()):
                uid = args.spk_tag + "_" + key[:-4]
                text = js[key]["clean"].upper()
                if args.lang_tag is None:
                    line = "%s %s\n" % (uid, text)
                else:
                    line = "%s <%s>%s\n" % (uid, args.lang_tag, text)
                out.write(line)


if __name__ == "__main__":
    main()
