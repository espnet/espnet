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
