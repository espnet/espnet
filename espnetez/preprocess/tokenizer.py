"""Convenience wrapper for ESPnet's text tokenizer.

This module exposes a single public function, :func:`tokenize`, that
constructs the command-line arguments required by
``espnet2.bin.tokenize_text`` and executes the tokenizer programmatically.
It is intended for use in Python scripts, unit tests, or CI pipelines
where invoking the ESPnet tokenizer as a subprocess is undesirable.

The wrapper forwards all keyword arguments to the underlying
`run_tokenizer` function after building the appropriate list of
arguments.  In addition to the standard tokenizer options it supports
the following convenience flags:

* ``write_vocabulary`` - whether to write a vocabulary file.
* ``blank``, ``oov`` and ``sos_eos`` - custom special symbols that will
  be added to the tokenizer's symbol table.

Because the arguments are passed to ``get_parser`` internally, any
additional options recognised by ESPnet can be supplied via
``**kwargs`` and will be parsed automatically.

Typical usage::

    from this_module import tokenize

    # Tokenise an English text file, generating a vocabulary.
    tokenize(
        input="raw.txt",
        output="tokenized.txt",
        write_vocabulary=True,
        blank="<blank>",
        oov="<unk>",
        sos_eos="<sos/eos>"
    )

The module relies on ``espnet2.bin.tokenize_text`` for the actual
tokenisation logic.  No files are returned; the function writes the
tokenised output and (optionally) the vocabulary directly to disk.

Functions
---------
tokenize(
    input,
    output,
    write_vocabulary=True,
    blank="<blank>",
    oov="<unk>",
    sos_eos="<sos/eos>",
     **kwargs)
    Tokenise *input* and write the result to *output*.

"""

from espnet2.bin.tokenize_text import get_parser
from espnet2.bin.tokenize_text import tokenize as run_tokenizer


def tokenize(
    input,
    output,
    write_vocabulary=True,
    blank="<blank>",
    oov="<unk>",
    sos_eos="<sos/eos>",
    **kwargs,
):
    """Tokenize the input text and saves the output to a specified file.

    This function utilizes the ESPnet tokenizer to process a given input text file,
    tokenize its contents, and save the results to an output file. Additionally,
    it can optionally write a vocabulary file and allows for the inclusion of
    special symbols such as blank tokens, out-of-vocabulary tokens, and start/end
    of sentence tokens.

    Args:
        input (str): Path to the input text file to be tokenized.
        output (str): Path to the output file where the tokenized text will be saved.
        write_vocabulary (bool, optional): Whether to write the vocabulary to a file.
            Defaults to True.
        blank (str, optional): Symbol to represent blank tokens. Defaults to "<blank>".
        oov (str, optional): Symbol to represent out-of-vocabulary tokens.
            Defaults to "<unk>".
        sos_eos (str, optional): Symbol to represent start and end of sentence tokens.
            Defaults to "<sos/eos>".
        **kwargs: Additional keyword arguments to customize the tokenizer behavior.

    Returns:
        None: The function does not return any value but writes the tokenized
        output to the specified file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the provided arguments are invalid.

    Examples:
        # Basic usage
        tokenize('input.txt', 'output.txt')

        # Customizing the tokenizer with additional symbols
        tokenize('input.txt', 'output.txt', write_vocabulary=False,
                 blank="<space>", oov="<unknown>", sos_eos="<start/end>")
    """
    parser = get_parser()
    kws = ["--input", input, "--output", output]
    kws += [
        "--write_vocabulary",
        str(write_vocabulary),
        "--add_symbol",
        f"{blank}:0",
        "--add_symbol",
        f"{oov}:1",
        "--add_symbol",
        f"{sos_eos}:-1",
    ]
    for k, v in kwargs.items():
        kws += [f"--{k}", v]

    args = parser.parse_args(kws)
    kwargs = vars(args)
    run_tokenizer(**kwargs)
