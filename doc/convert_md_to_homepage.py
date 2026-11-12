import copy
import re
from glob import glob

import configargparse
import markdown


def get_parser():
    parser = configargparse.ArgumentParser(
        description="Convert custom tags to markdown",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=str,
        help="source markdown file",
    )
    return parser


def small_bracket(text):
    # forward_core(nnet_output, ys, hlens, ylens)
    # -> forward_core<span class='small-bracket'>(nnet_output, ys, hlens, ylens)</span>"
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # Match the first top-level parenthesis group (allowing one level of
    # nesting), skipping one immediately preceded by "]" since that is a
    # markdown link's url, e.g. "[Field](../contract/Field.md#...)", not an
    # argument list to wrap.
    match = re.search(r"(?<!\])\(((?:[^()]|\([^()]*\))*)\)", text)
    if match:
        text = text.replace(
            match.group(0),
            f"<span class='small-bracket'>{match.group(0)}</span>",
            1,
        )
    return text


def convert(markdown_text):
    # convert "#### Examples" to :::note ::: block
    # We assume that this example block will continue to the next header.
    example_pattern = r"###\sExamples"
    _result = copy.copy(markdown_text)
    for match in re.finditer(example_pattern, _result):
        _result = _result.replace(match.group(0), "##### Examples")

    # convert ### to div with specific class
    h3_pattern = re.compile(r"^###\s+(.+)$", re.MULTILINE)
    for match in re.finditer(h3_pattern, _result):
        tag_removed = match.group(0).replace("### ", "")
        tag_removed = small_bracket(tag_removed)
        tag_removed = markdown.markdown(tag_removed)
        _result = _result.replace(
            match.group(0), f"\n<div class='custom-h3'>{tag_removed}</div>\n\n"
        )

    # convert "#### Note" to :::note ::: block
    # We assume that this note block will continue to the next header.
    note_pattern = r"####\sNOTE"
    for match in re.finditer(note_pattern, _result):
        _result = _result.replace(match.group(0), "##### NOTE")

    # Convert "####" to custom-h4 tag.
    h4_pattern = re.compile(r"^####\s+(.+)$", re.MULTILINE)
    for match in re.finditer(h4_pattern, _result):
        tag_removed = match.group(0).replace("#### ", "")
        tag_removed = small_bracket(tag_removed)
        tag_removed = markdown.markdown(tag_removed)
        _result = _result.replace(
            match.group(0), f"\n<div class='custom-h4'>{tag_removed}</div>\n\n"
        )
    return _result


if __name__ == "__main__":
    # parser
    args = get_parser().parse_args()

    for md in glob(f"{args.root}/**/*.md", recursive=True):
        markdown_text = open(md, "r").read()
        _result = convert(markdown_text)
        with open(md, "w") as f:
            f.write(_result)
