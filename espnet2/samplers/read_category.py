from typing import Dict
from typing import List

from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text


def get_category2utt(
    keys: List[str],
    utt2category_file: str = None,
) -> Dict[str, List]:
    """Read the utt2category file.
    Examples:
        utt2category:
            key1 category1
            key2 category2
        >>> get_category2utt('utt2category')
        {'category1': ['key1'], 'category2': ['key2']}
    """
    assert check_argument_types()

    category2utt = {}
    if utt2category_file is not None:
        utt2category = read_2column_text(utt2category_file)
        if set(utt2category) != set(keys):
            raise RuntimeError(
                "keys are mismatched between "
                f"{utt2category_file} != utterance keys"
            )
        for k in keys:
            category2utt.setdefault(utt2category[k], []).append(k)
    else:
        category2utt["default_category"] = keys
    return category2utt
