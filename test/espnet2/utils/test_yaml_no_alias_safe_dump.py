import pytest
import yaml

from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump

d = {"a": (1, 2, 3)}


@pytest.mark.parametrize(
    "data, desired",
    [(d, {"a": [1, 2, 3]}), ((d, d["a"]), [{"a": [1, 2, 3]}, [1, 2, 3]])],
)
def test_yaml_no_alias_safe_dump(data, desired):
    assert yaml.safe_load(yaml_no_alias_safe_dump(data)) == desired
