import argparse

# Note: punctuations are explicitly pronounced in WSJ dataset
symbol_map = {
    "\\-HYPHEN": "HYPHEN -",
    "\\,COMMA": "COMMA ,",
    "\\.PERIOD": "PERIOD .",
    '\\"DOUBLE\\-QUOTE': 'DOUBLE-QUOTE "',
    '\\"DOUBLE-QUOTE': 'DOUBLE-QUOTE "',
    "\\?QUESTION\\-MARK": "QUESTION-MARK ?",
    "\\?QUESTION-MARK": "QUESTION-MARK ?",
    "\\-\\-DASH": "DASH - -",
    "\\(LEFT\\-PAREN": "LEFT-PAREN (",
    "\\(LEFT-PAREN": "LEFT-PAREN (",
    "\\)RIGHT\\-PAREN": "RIGH-PAREN )",
    "\\)RIGHT-PAREN": "RIGH-PAREN )",
    "\\%PERCENT": "PERCENT %",
    "\\&AMPERSAND": "AMPERSAND &",
    "\\;SEMI\\-COLON": "SEMI-COLON ;",
    "\\{LEFT\\-BRACE": "LEFT-BRACE {",
    "\\{LEFT-BRACE": "LEFT-BRACE {",
    "\\}RIGHT\\-BRACE": "RIGHT-BRACE }",
    "\\}RIGHT-BRACE": "RIGHT-BRACE }",
    "\\!EXCLAMATION\\-POINT": "EXCLAMATION-POINT !",
    "\\!EXCLAMATION-POINT": "EXCLAMATION-POINT !",
    "\\'SINGLE\\-QUOTE": "SINGLE-QUOTE '",
    "\\/SLASH": "SLASH /",
    "\\:COLON": "COLON :",
    "\\;SEMI-COLON": "SEMI-COLON ;",
    '\\"QUOTE': 'QUOTE "',
    '\\"UNQUOTE': 'UNQUOTE "',
    "\\(PARENTHESES": "PARENTHESES (",
    "\\)UN\\-PARENTHESES": "UN-PARENTHESES )",
    "\\ ": " ",
    "\\(PAREN": "PAREN (",
    "\\)PAREN": "PAREN )",
    "\\)END\\-OF\\-PAREN": "END-OF-PAREN )",
    '\\"END\\-OF\\-QUOTE': 'END-OF-QUOTE "',
    "\\)END\\-THE\\-PAREN": "END-THE-PAREN )",
    '\\"END-OF-QUOTE': 'END-OF-QUOTE "',
    '\\"END\\-QUOTE': 'END-QUOTE "',
    "\\)CLOSE\\-PAREN": "CLOSE-PAREN )",
    "\\`": "`",
    '\\"CLOSE\\-QUOTE': 'CLOSE-QUOTE "',
    "\\(BRACE": "BRACE (",
    "\\)CLOSE\\-BRACE": "CLOSE-BRACE )",
    '\\"IN\\-QUOTES': 'IN-QUOTES "',
    "\\(BEGIN\\-PARENS": "BEGIN-PARENS (",
    "\\)END\\-PARENS": "END-PARENS )",
    "\\;": ";",
    "\\(IN\\-PARENTHESIS": "IN-PARENTHESIS (",
    "\\)CLOSE_PAREN": "CLOSE_PAREN )",
    "\\.POINT": "POINT .",
    "\\.\\.\\.": "...",
    "\\'": "'",
    "\\.'": ".'",
    "\\.": ".",
}


def main():
    parser = argparse.ArgumentParser(
        description='Create waves list from "wav.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="input transcription file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="output transcription file",
    )

    args = parser.parse_args()

    writer = open(args.output, "w", encoding="utf-8")
    for line in open(args.input, encoding="utf-8"):
        for old, new in symbol_map.items():
            line = line.replace(old, new)
        line = (
            " ".join(
                [
                    w
                    for w in line.strip().split()
                    if not (w.startswith("[") or w.endswith("]"))
                ]
            )
            + "\n"
        )
        writer.write(line)


if __name__ == "__main__":
    main()
