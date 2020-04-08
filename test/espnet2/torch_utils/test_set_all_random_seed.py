from espnet2.torch_utils.set_all_random_seed import set_all_random_seed


def test_set_all_random_seed():
    set_all_random_seed(0)
