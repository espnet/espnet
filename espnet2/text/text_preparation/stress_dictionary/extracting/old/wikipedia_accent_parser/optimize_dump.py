#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


ACCENT = u'́'


def optimize_dump(input_filepath, output_filepath):
    inside_page = False
    title = None
    accented_items = []
    write_comma = False

    i = 0

    with open(input_filepath, encoding='utf-8') as input_file:
        with open(output_filepath, 'w+', encoding='utf-8') as output_file:

            output_file.write("[")

            for line in input_file:
                if not inside_page:
                    if '<page' in line:
                        i += 1
                        if i % 1000 == 0:
                            print(i)
                        inside_page = True

                    continue

                if '</page>' in line:

                    if len(accented_items) > 0:
                        if write_comma:
                            output_file.write(",")
                        else:
                            write_comma = True

                        # print(accented_items, len(accented_items))

                        output_file.write(json.dumps({
                            "title": title,
                            "accented_items": accented_items
                        }, ensure_ascii=False))

                    inside_page = False
                    accented_items = []
                    continue

                if '<title>' in line and '</title>' in line:
                    title = line.split('<title>')[1].split('</title>')[0]
                    continue

                if "'''" in line:
                    item = line.split("'''")[1]
                    if ACCENT in item:
                        accented_items.append(item)
                    continue

            output_file.write("]")
