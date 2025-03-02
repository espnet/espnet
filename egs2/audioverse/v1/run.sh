#!/usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
log "$0 $*"
run_args=$(pyscripts/utils/print_args.py $0 "$@")

# Define default values for command-line options
parallel=false # Run recipes in parallel
wait_time=5  # Default wait time between parallel runs in seconds
filter_recipe=none # Filter recipes by name. Default none
recipe="" # Specific recipes to run, takes precedence. Default is all
config_prefix="" # Template config prefix tag
run_name="" # Name for this benchmark run
template_args="" # Arguments to replace in the template config
recipe_args="" # Arguments to pass to the recipe runners

# Help message
help_message=$(cat << EOF
Usage: $0 [OPTIONS] [ARGS_FOR_RECIPES...]
Run multiple recipes under Audioverse.

Options:
  # General
  --parallel BOOL           Run recipes in parallel (default: ${parallel})
  --wait_time SECONDS       Wait time between parallel runs (default: ${wait_time})

  # Recipe selection
  --filter_recipe LIST      Comma-separated list to run all recipes except these. Default is None.
  --recipe LIST             Comma-separated list of specific recipes to run. Default is all.

  # Config
  --config_prefix STRING    Prefix for the template config that should be used. The config path is expected to be
                            conf/template/{config_prefix}_{recipe_name}.yaml
  --run_name STRING         Name for this benchmark run (default: timestamp)
  --template_args LIST      Arguments to replace in the template config. Comma-separated list of KEY:value pairs.
                            For example: --template_args "BEATS_CHECKPOINT_PATH:/path/to/checkpoint"
                            Note that the KEY should be in ALLCAPS.
  --recipe_args LIST        Arguments to pass to the recipe runners. These are passed as-is to the recipe runners.
                            For example: --recipe_args "--stage 4 --batch_bins 800000"
  --help                    Display this help message
EOF
)


. utils/parse_options.sh
. ./path.sh

if [ -z "$config_prefix" ]; then
    log "Error: --config_prefix is required"
    exit 1
fi
if [ -z "$run_name" ]; then
    run_name=$(date "+%Y%m%d%H%M%S")
    log "Warning: --run_name is empty we will use timestamp: $run_name"
fi

exp_dir="exp/${run_name}"
mkdir -p "$exp_dir"
log "Created exp directory: $exp_dir"

# Register all recipe runners relative to this script
declare -A recipe_runners
recipe_runners["beans"]="../beans/cls1/run.sh"

recipe_runners["audioset20k"]="../as20k/cls1/run.sh"
recipe_runners["audioset2m"]="../as2m/cls1/run.sh"

recipe_runners["clotho_aqa"]="../clotho_v2/cls1/run_aqa_yn.sh,clotho_v2/cls1/run_aqa_open.sh"
recipe_runners["clotho_cle"]="../clotho_v2/cls1/run_entailment.sh"
recipe_runners["clotho_aac"]="../clotho_v2/asr1/run.sh"

generate_recipe_list() {
    local recipes_to_run=()

    # Add all recipes except filtered ones
    IFS=',' read -r -a filter_array <<< "$filter_recipe"
    for r in "${!recipe_runners[@]}"; do
        should_add=true
        for f in "${filter_array[@]}"; do
            if [[ -n "$f" && $r == *"$f"* ]]; then
                should_add=false
                break
            fi
        done
        [[ $should_add == true ]] && recipes_to_run+=("$r")
    done

    if [ -n "$recipe" ]; then
        recipes_to_run=() # Reset
        IFS=',' read -ra requested_recipes <<< "$recipe"
        for r in "${requested_recipes[@]}"; do
            if [[ -v recipe_runners[$r] ]]; then
                recipes_to_run+=("$r")
            else
                log "Unknown recipe: $r"
                return 1
            fi
        done
    fi

    if [ ${#recipes_to_run[@]} -eq 0 ]; then
        log "No recipes found matching criteria"
        return 1
    fi

    echo "${recipes_to_run[@]}"
    return 0
}

create_config() {
    local recipe=$1
    shift 1

    local template_dir="conf/template"
    local config_dir="conf/${run_name}"
    local template_path="${template_dir}/${config_prefix}_${recipe%.yaml}.yaml"
    local output_path="${config_dir}/${config_prefix}_${recipe%.yaml}.yaml"

    mkdir -p "$(dirname "$output_path")"
    if [ ! -f "$template_path" ]; then
        log "Template not found: $template_path"
        return 1
    fi

    # replace ARGUMENT placeholders
    cp "$template_path" "$output_path"
    for var_name in "${!template_map[@]}"; do
        if [[ -n "${template_map[$var_name]}" ]]; then
            sed -i "s|${var_name}|${template_map[$var_name]}|g" "$output_path"
        else
            log "Variable $var_name is not set"
        fi
    done

    log "Created config file: $output_path"
    return 0
}

construct_command_args() {
    # Only called after create_config
    local recipe=$1
    local runner=$2
    local config_dir="conf/${run_name}"
    local config_path="${config_dir}/${config_prefix}_${recipe%.yaml}.yaml"
    local recipe_dir="$(dirname $runner)"

    # if [ ! -d "$recipe_dir" ]; then
    #     log "Error: Recipe directory not found: $recipe_dir"
    #     return 1
    # fi

    task_name=$(basename $recipe_dir)
    if [[ $task_name == "cls1" ]]; then
        task_name="cls"
    elif [[ $task_name == "asr1" ]]; then
        task_name="asr"
    else
        log "Error: Unknown task: $task_name"
        return 1
    fi
    cmd_args="--${task_name}_config ${config_path} --${task_name}_tag ${run_name}"

    echo "$cmd_args"
    return 0
}

run_recipe() {
    local recipe=$1
    local runners="${recipe_runners[$recipe]}"
    shift

    local success=true
    local recipe_log="${exp_dir}/${recipe}.log"

    log "Processing recipe: $recipe"
    IFS=',' read -ra runner_list <<< "$runners"

    for runner in "${runner_list[@]}"; do
        log "Setting up task: $runner for recipe: $recipe"

        # Create config from the template
        create_config "$recipe" "$runner" || {
            log "Error: Failed to create config for $recipe|$runner"
            success=false
            continue
        }

        # Construct command for running the recipe
        local cmd_args=$(construct_command_args "$recipe" "$runner") || {
            log "Error: Failed to construct command for $recipe|$runner"
            success=false
            continue
        }

        log "Running command: $runner with args: ${cmd_args} ${recipe_args[@]}"
        log "Output will be saved to: $recipe_log"

        if [ "$parallel" = true ]; then
            (cd "$(dirname $runner)" && "./$(basename $runner) ${cmd_args} ${recipe_args[@]}" 2>&1 | tee -a "$recipe_log") &
            log "Started $recipe|$runner in background (PID: $!)"
            sleep "$wait_time"
        else
            (cd "$(dirname $runner)" && "./$(basename $runner) ${cmd_args} ${recipe_args[@]}" 2>&1 | tee -a "$recipe_log") || {
                log "Error: Recipe $recipe|$runner failed"
                success=false
            }
        fi
    done

    if $success; then
        return 0
    else
        return 1
    fi
}


summary_file="${exp_dir}/summary.txt"
echo "Benchmark Run: $run_name" > "$summary_file"
echo "Started: $(date)" >> "$summary_file"
echo "----------------------------------------" >> "$summary_file"

recipes=$(generate_recipe_list) || {
    log "Error: Failed to generate recipe list. Check --recipe and --filter_recipe arguments."
    exit 1
}
log "Recipe list is [ ${recipes[@]} ]"

total=0
successful=0
failed=0

# Run them all
for recipe in $recipes; do
    total=$((total + 1))
    if run_recipe "$recipe" "$run_args"; then
        successful=$((successful + 1))
        echo "$recipe: SUCCESS" >> "$summary_file"
    else
        failed=$((failed + 1))
        echo "$recipe: FAILED" >> "$summary_file"
    fi
done

# If running in parallel, wait for all background jobs
if [ "$parallel" = true ]; then
    log "Waiting for all recipes to complete..."
    wait

    # For parallel mode, we need to check logs for success/failure
    total=0
    successful=0
    failed=0

    for recipe in $recipes; do
        total=$((total + 1))
        recipe_log="${exp_dir}/${recipe}.log"
        if grep -q "Error\|Failed\|fatal" "$recipe_log"; then # TODO(shikhar): Parse!
            failed=$((failed + 1))
            echo "$recipe: FAILED (parallel run)" >> "$summary_file"
        else
            successful=$((successful + 1))
            echo "$recipe: SUCCESS (parallel run)" >> "$summary_file"
        fi
    done
fi

# Update summary
echo "----------------------------------------" >> "$summary_file"
echo "Finished: $(date)" >> "$summary_file"
echo "Results: $successful succeeded, $failed failed, $total total" >> "$summary_file"

log "All recipes completed. Summary in $summary_file"
log "Results: $successful succeeded, $failed failed, $total total"
