from espnet2.bin.launch import get_parser


def test_get_parser_max_num_log_files_default():
    args = get_parser().parse_args(["echo"])
    assert args.max_num_log_files == 1000


def test_get_parser_max_num_log_files_is_int():
    args = get_parser().parse_args(["--max_num_log_files", "2", "echo"])
    assert args.max_num_log_files == 2
