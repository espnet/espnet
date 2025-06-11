import yaml


class NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        """Disable anchor/alias in yaml because looks ugly."""
        return True


def yaml_no_alias_safe_dump(data, stream=None, **kwargs):
    """Safe-dump in yaml with no anchor/alias."""
    return yaml.dump(
        data, stream, allow_unicode=True, Dumper=NoAliasSafeDumper, **kwargs
    )
