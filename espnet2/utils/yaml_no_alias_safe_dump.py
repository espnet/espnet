import yaml


class NoAliasSafeDumper(yaml.SafeDumper):
    """
        NoAliasSafeDumper is a custom YAML dumper that disables the use of anchors and
    aliases in the dumped YAML data. This can help maintain a cleaner and more
    readable output by preventing the creation of references that can complicate
    the structure of the YAML document.

    Attributes:
        None

    Methods:
        ignore_aliases(data):
            Override the ignore_aliases method to always return True, disabling
            anchor and alias creation for the given data.

    Examples:
        >>> data = {'key1': 'value1', 'key2': ['value2a', 'value2b']}
        >>> yaml_str = yaml_no_alias_safe_dump(data)
        >>> print(yaml_str)
        key1: value1
        key2:
        - value2a
        - value2b

    Note:
        This class is a subclass of yaml.SafeDumper, which provides a safe way to
        dump YAML data without allowing arbitrary code execution.
    """

    # Disable anchor/alias in yaml because looks ugly
    def ignore_aliases(self, data):
        """
            Disable the creation of YAML aliases for the given data.

        This method is overridden to ensure that no anchors or aliases are
        created when dumping data to YAML. This is particularly useful when
        you want a clean representation of your data without the potential
        confusion of YAML aliases.

        Args:
            data: The data to be serialized into YAML.

        Returns:
            bool: Always returns True, indicating that aliases should be
            ignored.

        Examples:
            >>> dumper = NoAliasSafeDumper()
            >>> dumper.ignore_aliases({"key": "value"})
            True
        """
        return True


def yaml_no_alias_safe_dump(data, stream=None, **kwargs):
    """
        Safe-dump in YAML format without using anchors or aliases.

    This function utilizes the `NoAliasSafeDumper` to ensure that the output YAML
    does not include any anchors or aliases, which can sometimes make the output
    less readable. This is particularly useful when the YAML output is intended
    for presentation or logging purposes.

    Args:
        data (Any): The data to be serialized to YAML format.
        stream (file-like object, optional): The stream to which the YAML data
            will be written. If not provided, the function returns the YAML as a
            string.
        **kwargs: Additional keyword arguments to be passed to the `yaml.dump`
            function.

    Returns:
        str or None: Returns the YAML representation of the data as a string if
        `stream` is not provided. Otherwise, returns None.

    Examples:
        >>> yaml_no_alias_safe_dump({'name': 'John', 'age': 30})
        'age: 30\nname: John\n'

        >>> with open('output.yaml', 'w') as f:
        ...     yaml_no_alias_safe_dump({'name': 'John', 'age': 30}, stream=f)

    Note:
        This function requires the PyYAML library to be installed.

    Raises:
        yaml.YAMLError: If an error occurs during the YAML serialization process.
    """
    return yaml.dump(
        data, stream, allow_unicode=True, Dumper=NoAliasSafeDumper, **kwargs
    )
