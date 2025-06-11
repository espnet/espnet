"""YAML no alias safe dump.

This module provides a custom YAML SafeDumper that disables the use of anchors
and aliases when dumping Python objects to YAML. By overriding the
`ignore_aliases` method in the `NoAliasSafeDumper` class, all anchors (&) and
aliases (*) are suppressed in the output, resulting in cleaner and more readable YAML.

Functions:
    yaml_no_alias_safe_dump(data, stream=None, **kwargs):
        Dumps the given data to YAML format using the custom dumper that
        disables anchors and aliases. This function ensures safe dumping with Unicode
        support by default.

Classes:
    NoAliasSafeDumper(yaml.SafeDumper):
        A subclass of PyYAML's SafeDumper that overrides alias handling to always
        return True, effectively disabling YAML anchors and aliases in the output.
"""

import yaml


class NoAliasSafeDumper(yaml.SafeDumper):
    """A custom YAML SafeDumper that disables the use of anchors and aliases.

    This dumper overrides the `ignore_aliases` method to always return True,
    ensuring that YAML output does not contain anchors (&) or aliases (*),
    which can make the output less readable or "ugly" in certain contexts.

    Example usage:
        yaml.dump(data, Dumper=NoAliasSafeDumper)
    """

    def ignore_aliases(self, data):
        """Disable anchor/alias in yaml because looks ugly."""
        return True


def yaml_no_alias_safe_dump(data, stream=None, **kwargs):
    """Safe-dump in yaml with no anchor/alias."""
    return yaml.dump(
        data, stream, allow_unicode=True, Dumper=NoAliasSafeDumper, **kwargs
    )
