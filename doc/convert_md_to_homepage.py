import markdown
import re
import copy
import configargparse
from glob import glob


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
    brackets = re.findall(r'\((.*)\)', text)
    if len(brackets) > 0:
        text = text.replace(
            f"({brackets[0]})",
            f"<span class='small-bracket'>({brackets[0]})</span>"
        )
    return text


def convert(markdown_text):
    # convert "#### Examples" to :::note ::: block
    # We assume that this example block will continue to the next header.
    example_pattern = r'###\sExamples'
    _result = copy.copy(markdown_text)
    for match in re.finditer(example_pattern, _result):
        _result = _result.replace(
            match.group(0),
            "##### Examples"
        )

    # convert ### to div with specific class
    h3_pattern = re.compile(r'^###\s+(.+)$', re.MULTILINE)
    for match in re.finditer(h3_pattern, _result):
        tag_removed = match.group(0).replace("### ", "")
        tag_removed = small_bracket(tag_removed)
        tag_removed = markdown.markdown(tag_removed)
        _result = _result.replace(
            match.group(0),
            f"<div class='custom-h3'>{tag_removed}</div>\n"
        )

    # convert "#### Note" to :::note ::: block
    # We assume that this note block will continue to the next header.
    note_pattern = r'####\sNOTE'
    for match in re.finditer(note_pattern, _result):
        _result = _result.replace(
            match.group(0),
            "##### NOTE"
        )

    # Convert "####" to custom-h4 tag.
    h4_pattern = re.compile(r'^####\s+(.+)$', re.MULTILINE)
    for match in re.finditer(h4_pattern, _result):
        tag_removed = match.group(0).replace("#### ", "")
        tag_removed = small_bracket(tag_removed)
        tag_removed = markdown.markdown(tag_removed)
        _result = _result.replace(
            match.group(0),
            f"<div class='custom-h4'>{tag_removed}</div>\n"
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
