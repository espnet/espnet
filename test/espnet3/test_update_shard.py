from espnet3.trainer.dataloader import update_shard

# ===============================================================
# Test Case Summary for update_shard
# ===============================================================
#
# String Replacement Logic
# | Test Name                           | Description                                                                 | # noqa: E501
# |------------------------------------|-----------------------------------------------------------------------------| # noqa: E501
# | test_update_shard_in_dict          | Replaces `{shard_idx}` in string values inside a dictionary                | # noqa: E501
# | test_update_shard_in_list          | Replaces `{shard_idx}` in strings inside a list                            | # noqa: E501
# | test_update_shard_multiple_occurrences | Replaces multiple `{shard_idx}` within a single string                 | # noqa: E501
# | test_update_shard_multiple_locations  | Replaces `{shard_idx}` in various nested structures across the config     | # noqa: E501
#
# Nesting & Structural Integrity
# | Test Name                           | Description                                                                 | # noqa: E501
# |------------------------------------|-----------------------------------------------------------------------------| # noqa: E501
# | test_update_shard_nested_mix       | Handles nested dict + list structures with placeholder substitutions       | # noqa: E501
# | test_update_shard_empty_structures | Gracefully handles empty lists and dicts                                   | # noqa: E501
#
# Non-modifying / Edge Cases
# | Test Name                           | Description                                                                 | # noqa: E501
# |------------------------------------|-----------------------------------------------------------------------------| # noqa: E501
# | test_update_shard_no_change        | Leaves strings unchanged if no `{shard_idx}` is present                    | # noqa: E501
# | test_update_shard_non_string       | Leaves non-string values (int, float, bool, None) untouched                | # noqa: E501


def test_update_shard_in_dict():
    config = {"path": "data/shard_{shard_idx}/input.txt"}
    result = update_shard(config, 2)
    assert result == {"path": "data/shard_2/input.txt"}


def test_update_shard_in_list():
    config = ["stats/shard_{shard_idx}/shape", "other/file.txt"]
    result = update_shard(config, 5)
    assert result == ["stats/shard_5/shape", "other/file.txt"]


def test_update_shard_nested_mix():
    config = {
        "data": {"input": ["file_{shard_idx}.txt", {"meta": "meta_{shard_idx}.json"}]}
    }
    result = update_shard(config, 3)
    assert result == {"data": {"input": ["file_3.txt", {"meta": "meta_3.json"}]}}


def test_update_shard_no_change():
    config = {"unchanged": "no_placeholder_here.txt"}
    result = update_shard(config, 0)
    assert result == config


def test_update_shard_non_string():
    config = {"int_val": 123, "bool_val": True, "none_val": None, "float_val": 3.14}
    result = update_shard(config, 1)
    assert result == config


def test_update_shard_empty_structures():
    config = {"empty_list": [], "empty_dict": {}}
    result = update_shard(config, 9)
    assert result == {"empty_list": [], "empty_dict": {}}


def test_update_shard_multiple_occurrences():
    config = {"file": "out_{shard_idx}_v{shard_idx}.log"}
    result = update_shard(config, 7)
    assert result == {"file": "out_7_v7.log"}


def test_update_shard_multiple_locations():
    config = {
        "input_path": "data/shard_{shard_idx}/input.txt",
        "meta": {
            "log_dir": "logs/run_{shard_idx}/",
            "settings": ["checkpoint_{shard_idx}.pt", "summary.txt"],
        },
        "unchanged": 123,
        "list_of_dicts": [{"file": "out_{shard_idx}.json"}, {"note": "no change here"}],
    }

    shard_idx = 42
    updated = update_shard(config, shard_idx)

    expected = {
        "input_path": "data/shard_42/input.txt",
        "meta": {
            "log_dir": "logs/run_42/",
            "settings": ["checkpoint_42.pt", "summary.txt"],
        },
        "unchanged": 123,
        "list_of_dicts": [{"file": "out_42.json"}, {"note": "no change here"}],
    }

    assert updated == expected
