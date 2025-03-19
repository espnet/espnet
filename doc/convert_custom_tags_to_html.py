import os  # noqa
import glob
import re
import configargparse

ALL_HTML_TAGS = [
    "a",
    "abbr",
    "acronym",
    "address",
    "applet",
    "area",
    "article",
    "aside",
    "audio",
    "b",
    "base",
    "basefont",
    "bdi",
    "bdo",
    "big",
    "blockquote",
    "body",
    "br",
    "button",
    "canvas",
    "caption",
    "center",
    "cite",
    "code",
    "col",
    "colgroup",
    "data",
    "datalist",
    "dd",
    "del",
    "details",
    "dfn",
    "dialog",
    "dir",
    "div",
    "dl",
    "dt",
    "em",
    "embed",
    "fieldset",
    "figcaption",
    "figure",
    "font",
    "footer",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hgroup",
    "hr",
    "html",
    "i",
    "iframe",
    "img",
    "input",
    "ins",
    "kbd",
    "label",
    "legend",
    "li",
    "link",
    "main",
    "map",
    "mark",
    "menu",
    "meta",
    "meter",
    "nav",
    "noframes",
    "noscript",
    "object",
    "ol",
    "optgroup",
    "option",
    "output",
    "p",
    "param",
    "picture",
    "pre",
    "progress",
    "q",
    "rp",
    "rt",
    "ruby",
    "s",
    "samp",
    "script",
    "search",
    "section",
    "select",
    "small",
    "source",
    "span",
    "strike",
    "strong",
    "style",
    "sub",
    "summary",
    "sup",
    "svg",
    "table",
    "tbody",
    "td",
    "template",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "time",
    "title",
    "tr",
    "track",
    "tt",
    "u",
    "ul",
    "var",
    "video",
    "wbr",
]

LANGUAGE_TAG_SET = [
    ("default", "text"),
    ("pycon", "python"),
    ("cd", "text"),
]


def get_parser():
    parser = configargparse.ArgumentParser(
        description="Convert custom tags to markdown",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=str,
        help="source python files that contain get_parser() func",
    )
    return parser


def replace_custom_tags(content):
    # Regex to find tags and their content
    tag_pattern = re.compile(r'<(?!!--)([^>]+)>')

    def replace_tag(match):
        tag_name = match.group(1)
        if len(tag_name) > 50:
            # heuristics to ignore tags with too long names
            # This might occur with image tags, since they have image data
            # in base64 format.
            return match.group(0)

        if (
            tag_name.split()[0] not in ALL_HTML_TAGS
            or (
                len(tag_name.split()) > 1 and "=" not in tag_name
            )
        ):
            return f"&lt;{tag_name}&gt;"

        end_tag_pattern = re.compile(f'</{tag_name.split()[0]}>')
        end_tag_match = end_tag_pattern.search(content, match.end())
        if not end_tag_match:
            return f"&lt;{tag_name}&gt;"
        return match.group(0)
    return tag_pattern.sub(replace_tag, content)


def replace_string_tags(content):
    # Regex to find tags and their content
    tag_pattern = re.compile(r"['|\"]<(?!\/)(.+?)(?!\/)>['|\"]")

    def replace_tag(match):
        tag_name = match.group(1)
        if len(tag_name) > 50:
            # heuristics to ignore tags with too long names
            # This might occur with image tags, since they have image data
            # in base64 format.
            return match.group(0)
        if (
            tag_name.split()[0] not in ALL_HTML_TAGS
            or (
                len(tag_name.split()) > 1 and "=" not in tag_name
            )
        ):
            return f"'&lt;{tag_name}&gt;'"

        end_tag_pattern = re.compile(f'</{tag_name.split()[0]}>')
        end_tag_match = end_tag_pattern.search(content, match.end())
        if not end_tag_match:
            return f"'&lt;{tag_name}&gt;'"
        return match.group(0)
    return tag_pattern.sub(replace_tag, content)


def replace_language_tags(content):
    for (label, lang) in LANGUAGE_TAG_SET:
        content = content.replace(f"```{label}", f"```{lang}")

    return content


if __name__ == "__main__":
    # parser
    args = get_parser().parse_args()

    for md in glob.glob(f"{args.root}/*.md", recursive=True):
        with open(md, "r") as f:
            content = f.read()

        # Replace the "" and "" with "&lt;" and "&gt;", respectively
        # if the tag is not in ALL_HTML_TAGS and does not have its end tag
        # we need to apply this two functions because
        # there are custom tags like: "<custom-tag a='<type>' b='<value>' />"
        content = replace_language_tags(content)
        content = replace_string_tags(content)
        content = replace_custom_tags(content)

        with open(md, "w") as f:
            f.write(content)

    for md in glob.glob(f"{args.root}/**/*.md", recursive=True):
        with open(md, "r") as f:
            content = f.read()

        # Replace the "" and "" with "&lt;" and "&gt;", respectively
        # if the tag is not in ALL_HTML_TAGS
        content = replace_language_tags(content)
        content = replace_string_tags(content)
        content = replace_custom_tags(content)

        with open(md, "w") as f:
            f.write(content)
