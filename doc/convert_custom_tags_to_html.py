import re
from pathlib import Path

import configargparse

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


def replace_escaped_tags(content):
    # Convert author-marked literal angle brackets in prose.
    tag_pattern = re.compile(r"\\<([^<>\n]{1,50}?)(?:\\)?>")

    def replace_tag(match):
        tag_name = match.group(1)
        if len(tag_name) > 50:
            # heuristics to ignore tags with too long names
            # This might occur with image tags, since they have image data
            # in base64 format.
            return match.group(0)

        first_part = tag_name.split()[0]

        # Handle closing tags like </a>, </div>, etc.
        if first_part.startswith("/"):
            base_tag = first_part[1:]
            if base_tag in ALL_HTML_TAGS:
                return match.group(0)
            return f"&lt;{tag_name}&gt;"

        if first_part not in ALL_HTML_TAGS or (
            len(tag_name.split()) > 1 and "=" not in tag_name
        ):
            return f"&lt;{tag_name}&gt;"

        end_tag_pattern = re.compile(f"</{first_part}>")
        end_tag_match = end_tag_pattern.search(content, match.end())
        if not end_tag_match:
            return f"&lt;{tag_name}&gt;"
        return match.group(0)

    return tag_pattern.sub(replace_tag, content)


def replace_string_tags(content):
    # Convert tag-like string literals in generated HTML signatures.
    tag_pattern = re.compile(r"(?P<quote>['\"])<(?P<tag>[^<>\n]{1,50})>(?P=quote)")

    def replace_tag(match):
        quote = match.group("quote")
        tag_name = match.group("tag")
        return f"{quote}&lt;{tag_name}&gt;{quote}"

    return tag_pattern.sub(replace_tag, content)


def replace_all_tags(content):
    # Generated docstrings do not support raw HTML/Vue tags. Keep Sphinx's
    # leading metadata comments hidden, but escape every other tag-like span.
    tag_pattern = re.compile(r"<(?![!]--)([^<>\n]+?)>")

    def replace_tag(match):
        tag_name = match.group(1)
        return f"&lt;{tag_name}&gt;"

    return tag_pattern.sub(replace_tag, content)


def replace_language_tags(content):
    for label, lang in LANGUAGE_TAG_SET:
        content = content.replace(f"```{label}", f"```{lang}")

    return content


def _split_fenced_code_blocks(text):
    segments = []
    fence_re = re.compile(r"^[ \t]*([`~]{3,}).*$")
    in_fence = False
    fence_marker = None
    buffer = []

    for line in text.splitlines(keepends=True):
        if not in_fence:
            match = fence_re.match(line)
            if match:
                if buffer:
                    segments.append((False, "".join(buffer)))
                    buffer = []
                in_fence = True
                fence_marker = match.group(1)[0]
                buffer.append(line)
            else:
                buffer.append(line)
        else:
            buffer.append(line)
            match = fence_re.match(line)
            if match and match.group(1)[0] == fence_marker:
                segments.append((True, "".join(buffer)))
                buffer = []
                in_fence = False
                fence_marker = None

    if buffer:
        segments.append((in_fence, "".join(buffer)))

    return segments


def _split_inline_code(text):
    segments = []
    pattern = re.compile(r"(`+)([^`\n]*?)\1")
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            segments.append((False, text[last : match.start()]))
        segments.append((True, match.group(0)))
        last = match.end()
    if last < len(text):
        segments.append((False, text[last:]))
    return segments


def replace_tags_skip_code(content):
    return _replace_tags_skip_code(content, replace_string_tags, replace_escaped_tags)


def replace_all_tags_skip_code(content):
    return _replace_tags_skip_code(content, replace_all_tags)


def _replace_tags_skip_code(content, *replacers):
    content = replace_language_tags(content)
    fenced_segments = _split_fenced_code_blocks(content)
    output = []
    for is_code_block, block in fenced_segments:
        if is_code_block:
            output.append(block)
            continue
        inline_segments = _split_inline_code(block)
        for is_inline_code, seg in inline_segments:
            if is_inline_code:
                output.append(seg)
            else:
                for replacer in replacers:
                    seg = replacer(seg)
                output.append(seg)
    return "".join(output)


def is_build_path(root):
    return "build" in Path(root).parts


def iter_markdown_files(root):
    root_path = Path(root)
    if root_path.is_file():
        if root_path.suffix == ".md":
            yield root_path
        return

    yield from root_path.rglob("*.md")


if __name__ == "__main__":
    # parser
    args = get_parser().parse_args()
    if is_build_path(args.root):
        replace_tags = replace_all_tags_skip_code
    else:
        replace_tags = replace_tags_skip_code

    for md in iter_markdown_files(args.root):
        with open(md, "r", encoding="utf-8") as f:
            content = f.read()

        content = replace_tags(content)

        with open(md, "w", encoding="utf-8") as f:
            f.write(content)
