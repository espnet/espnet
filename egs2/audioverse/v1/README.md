# Audioverse Benchmark Runner

A unified framework for **running multiple audio processing recipes within ESPnet with configurable settings from a single command**.
This tool manages configuration templates, experiment directories, and execution parameters for reproducible benchmarking.

## Usage Examples

### Run all audio recipes with BEATs model

```bash
./run.sh --config_prefix beats
```

### Run with different settings across multiple recipes

```bash
# First experiment
./run.sh --config_prefix beats --run_name iter3 \
             --template_args "CHECKPOINT_PATH:/path/to/beatsiter3.pt"

# Second experiment    
./run.sh --config_prefix beats --run_name iter2 \
             --template_args "CHECKPOINT_PATH:/path/to/beatsiter2.pt"
```
This functionality can be used to change arguments between recipes by using template configs. For example, one config may not have the template key `CHECKPOINT_PATH`.

### Run two specific recipes in parallel with custom parameters (passed to all recipes)

```bash
./run.sh --config_prefix beats --recipe "cle_bert,aqa_yn_bert" \
             --parallel true --wait_time 10 \
             --recipe_args "--stage 2" \
             --run_name beats_stage2_run
```
Recipe args are passed to recipes directly and help in controlling stages etc.

### Test run without execution

```bash
./run.sh --config_prefix beats --dry_run --template_args "CHECKPOINT_PATH:/path/to/model"
```
The `--dry_run` option prints out the command that will be run.

## Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--parallel BOOL` | Run recipes in parallel | `false` |
| `--wait_time SECONDS` | Wait time between parallel runs | `5` |
| `--store_locally BOOL` | Store all data in current directory | `true` |
| `--dry_run` | Print commands without executing | `false` |
| `--filter_recipe LIST` | Comma-separated list of recipes to exclude | `none` |
| `--recipe LIST` | Comma-separated list of specific recipes to run | All recipes |
| `--config_prefix STRING` | Prefix for config templates | **Required** |
| `--run_name STRING` | Name for this benchmark run | Timestamp |
| `--template_args LIST` | KEY:value pairs to replace in template configs | None |
| `--recipe_args LIST` | Arguments passed to all recipe runners | None |

## Output and Logs

- Each recipe's output is logged to: `exp/{run_name}/{recipe}.log`
- Summary report generated at: `exp/{run_name}/summary.txt`
- Configs generated from the template are stored in: `conf/{run_name}/`

## Adding New Recipes

To add new recipes, update the `recipe_runners` associative array in the script:

```bash
recipe_runners["new_recipe_name"]="path/to/run_script.sh|--optional args"
```
Separate the path to recipe and the arguments by a `'|'`.

## Notes

- Always specify `--config_prefix` (required)
- Recipe naming convention requires unique names without dots
- Template placeholders in config files should be in ALLCAPS


## Features

### Template-based Configuration

- Uses YAML templates from `conf/template/{config_prefix}_{recipe_name}.yaml`
- Dynamically replaces placeholders using template arguments
- Example: `--template_args "BEATS_CHECKPOINT_PATH:/path/to/checkpoint,BATCH_SIZE:32"`

### Run Management

- Each benchmark run has a unique name (`--run_name`) or auto-generated timestamp
- Creates organized directory structure: `exp/{run_name}/`, `conf/{run_name}/`
- Different runs can use the same templates with different arguments
- Example:
  ```bash
  ./run.sh --run_name experiment1 --template_args "LR:0.001"
  ./run.sh --run_name experiment2 --template_args "LR:0.0001"
  ```

### Local Storage Organization

- With `--store_locally=true` (default), centralizes all data in current directory
- Organizes by recipe: `data/{recipe}/`, `dump/{recipe}/`, `exp/{recipe}/`
- Set `--store_locally=false` to use original paths from recipe scripts

### Recipe Selection

- Run specific recipes: `--recipe "cle_bert,aqa_yn_bert"`
- Exclude specific recipes: `--filter_recipe "clap"`
- All recipes run by default

### Execution Control

- Parallel execution: `--parallel true --wait_time 5`
- Simulation mode: `--dry_run` (prints commands without executing)
- Pass additional arguments to all recipes: `--recipe_args "--ngpu 2 --stage 3"`
