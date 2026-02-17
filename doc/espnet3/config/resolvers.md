---
title: ESPnet3 Config Resolvers
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Config Resolvers

ESPnet3 registers a small set of OmegaConf resolvers to pull external data into
YAML configs at load time. These are defined in `espnet3.utils.config` and are
available in all stage configs.

## ✅ At a glance

| Resolver | Description |
| --- | --- |
| `load_line` | Load lines from a text file into a list. |
| `load_yaml` | Load a YAML file or a specific key from it. |

## load_line

Use `load_line` to read a text file (one entry per line) and inject it into a
config. This is commonly used for vocab or token lists.

Sample file:

```
<blank>
<sos/eos>
<unk>
```

```yaml
token_list: ${load_line:conf/token_list.txt}
```

When the config is loaded, `token_list` becomes:

```yaml
token_list:
  - "<blank>"
  - "<sos/eos>"
  - "<unk>"
```

## load_yaml

Use `load_yaml` to load an entire YAML file or a single value from it.

Sample file (`conf/train.yaml`):

```yaml
exp_tag: asr_template_train
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

```yaml
exp_tag: ${load_yaml:conf/train.yaml,exp_tag}
full_cfg: ${load_yaml:conf/train.yaml}
```

When the config is loaded, the values are resolved from the referenced file.

Example of the resolved result in a second config:

```yaml
exp_tag: asr_template_train
full_cfg:
  exp_tag: asr_template_train
  exp_dir: ${recipe_dir}/exp/${exp_tag}
```

When a key is provided, it uses dot notation and raises an error if the key is
missing.
