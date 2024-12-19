import yaml

from espnetez.task import get_ez_task


def convert_none_to_None(dic):
    """
    Recursively convert string representations of 'none' in a dictionary to None.

    This function traverses a dictionary and replaces any occurrences of the string
    "none" with the actual Python None type. If a value in the dictionary is another
    dictionary, this function is called recursively to ensure all nested dictionaries
    are processed.

    Args:
        dic (dict): A dictionary potentially containing the string "none" as a value.

    Returns:
        dict: The input dictionary with all instances of the string "none" replaced
        by None.

    Examples:
        >>> sample_dict = {'key1': 'none', 'key2': {'subkey1': 'none',
            'subkey2': 'value'}}
        >>> convert_none_to_None(sample_dict)
        {'key1': None, 'key2': {'subkey1': None, 'subkey2': 'value'}}

        >>> nested_dict = {'level1': {'level2': {'level3': 'none'}}}
        >>> convert_none_to_None(nested_dict)
        {'level1': {'level2': {'level3': None}}}

    Note:
        This function modifies the input dictionary in place, but it also returns
        the modified dictionary for convenience.
    """
    for k, v in dic.items():
        if isinstance(v, dict):
            dic[k] = convert_none_to_None(dic[k])

        elif v == "none":
            dic[k] = None
    return dic


def from_yaml(task, path):
    """
    Load configuration from a YAML file and merge it with the default configuration.

    This function reads a YAML configuration file from the specified path and merges
    its contents with the default configuration for the specified task. If there are any
    keys in the YAML file that have the string value "none", they are converted to
    `None` type. The resulting configuration dictionary is returned.

    Args:
        task (str): The name of the task for which the configuration is being loaded.
        path (str): The file path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the merged configuration settings.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If the YAML file is not formatted correctly.

    Examples:
        >>> config = from_yaml('speech_recognition', 'config.yaml')
        >>> print(config)
        {'learning_rate': 0.001, 'batch_size': 32, 'preprocessor_conf': None}

        >>> config = from_yaml('text_to_speech', 'path/to/config.yaml')
        >>> print(config['model_type'])
        'tacotron2'

    Note:
        Ensure that the task name provided corresponds to a valid task class
        in the `espnetez.task` module to avoid runtime errors.
    """
    task_class = get_ez_task(task)
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # get default configuration from task class.
    default_config = task_class.get_default_config()
    default_config.update(config)

    default_config = convert_none_to_None(default_config)

    return default_config


def update_finetune_config(task, pretrain_config, path):
    """
    Update the fine-tuning configuration with values from a specified YAML file.

    This function loads the fine-tuning configuration from a YAML file and
    updates the provided pre-training configuration dictionary. It prioritizes
    values from the fine-tuning configuration, while ensuring that any
    distributed-related settings are reset to their defaults. Additionally,
    it integrates default configurations from the specified task.

    Args:
        task (str): The name of the task for which the configuration is being updated.
        pretrain_config (dict): The existing pre-training configuration dictionary
            to be updated.
        path (str): The file path to the YAML file containing the fine-tuning
            configuration.

    Returns:
        dict: The updated pre-training configuration dictionary after merging with the
            fine-tuning configuration and defaults from the specified task.

    Examples:
        >>> pretrain_cfg = {
        ...     "learning_rate": 0.001,
        ...     "batch_size": 32,
        ...     "dist_backend": "nccl"
        ... }
        >>> updated_cfg = update_finetune_config("asr", pretrain_cfg,
            "finetune_config.yaml")
        >>> print(updated_cfg)
        {
            "learning_rate": 0.0001,  # updated from finetune_config.yaml
            "batch_size": 32,
            "dist_backend": "nccl",
            "other_config": "default_value"  # from task defaults
        }

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If the YAML file is improperly formatted.

    Note:
        The function assumes that the task class provides a method
        `get_default_config()` which returns the default configuration as a dictionary.
    """
    with open(path, "r") as f:
        finetune_config = yaml.load(f, Loader=yaml.Loader)
    default_config = get_ez_task(task).get_default_config()

    # update pretrain_config with finetune_config
    # and update distributed related configs to the default.
    for k in list(pretrain_config):
        if "dist_" in k or "_rank" in k:
            pretrain_config[k] = default_config[k]
        elif k in finetune_config and pretrain_config[k] != finetune_config[k]:
            pretrain_config[k] = finetune_config[k]

    for k in list(default_config):
        if k not in pretrain_config:
            pretrain_config[k] = default_config[k]

    if "preprocessor_conf" in finetune_config:
        for k, v in finetune_config["preprocessor_conf"].items():
            pretrain_config["preprocessor_conf"][k] = v

    pretrain_config = convert_none_to_None(pretrain_config)

    return pretrain_config
