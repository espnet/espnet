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
store_locally=true # If true all runs will have data, dump and expdir in current audioverse folder. Otherwise we use the default values of recipe.
dry_run=false # Do not run just print commands
filter_recipe=none # Filter recipes by name. Default none
recipe="" # Specific recipes to run, takes precedence. Default is all
config_prefix="" # Template config prefix tag. This should be a directory inside conf/template.
run_name="" # Name for this benchmark run
template_args="" # Arguments to replace in the template config
recipe_args="" # Arguments to pass to the recipe runners
task_args="" # Arguments to pass as <task_name>_args
log_wandb=false # Log to wandb if true. Please provide wandb_entity in task_args if true.

# Help message
help_message=$(cat << EOF
Usage: $0 --config_prefix TEMPLATE_DIRECTORY [OPTIONS] [ARGS_FOR_RECIPES...]
Run multiple recipes under Audioverse.

Options:
  # General
  --parallel BOOL           Run recipes in parallel (default: ${parallel})
  --wait_time SECONDS       Wait time between parallel runs (default: ${wait_time})
  --store_locally BOOL      Store all recipe data in this folder if true (default: ${store_locally})
  --dry_run                 If true, just print commands (default: ${dry_run})

  # Recipe selection
  --filter_recipe LIST      Comma-separated list to run all recipes except these. Default is None.
  --recipe LIST             Comma-separated list of specific recipes to run. Default is all.

  # Config
  --config_prefix STRING    Prefix for the config template of this run. The config path is expected to be
                            conf/template/{config_prefix}/{recipe_name}.yaml
  --run_name STRING         Name for this benchmark run (default: timestamp)
  --template_args LIST      Arguments to replace in the template config. Comma-separated list of KEY:value pairs.
                            For example: --template_args "BEATS_CHECKPOINT_PATH:/path/to/checkpoint"
                            Note that the KEY should be in ALLCAPS.
  --recipe_args LIST        Arguments to pass to the recipe runners. These are passed as-is to the recipe runners.
                            For example: --recipe_args "--stage 4 --batch_bins 800000"
  --task_args LIST          Arguments to pass as <task_name>_args.
                            For example: --task_args "--wandb_project audioverse --wandb_entity shikhar"
  --log_wandb BOOL          Log to wandb if true. Please provide wandb_entity in task_args if true.
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

if ${log_wandb}; then
    if [[ -z "${task_args}" || "${task_args}" != *"--wandb_entity"* ]]; then
        log "Error: --task_args must contain --wandb_entity if --log_wandb is true"
        exit 1
    fi
fi

exp_dir="$(pwd)/exp/runs/${run_name}"
mkdir -p "$exp_dir"
log "Created exp directory: $exp_dir"

declare -A template_map
IFS=',' read -ra template_args_list <<< "$template_args"
for arg in "${template_args_list[@]}"; do
    IFS=':' read -ra arg_parts <<< "$arg"
    if [ ${#arg_parts[@]} -ne 2 ]; then
        log "Error: Invalid template argument: $arg"
        exit 1
    fi
    template_map[${arg_parts[0]}]=${arg_parts[1]}
done


# Register all recipe runners relative to this script
# Format: [recipe_name]="[path_to_script]|[additional_arguments]"
declare -A recipe_runners

# Captioning tasks
recipe_runners["audiocaps_aac"]="../../clotho_v2/asr1/run_pt.sh|"
recipe_runners["clotho_aac"]="../../clotho_v2/asr1/run_ft.sh|"

# BERT audio-text classification
recipe_runners["cle_bert"]="../../clotho_v2/cls1/run_entailment.sh|--hugging_face_model_name_or_path bert-base-uncased"
recipe_runners["aqa_yn_bert"]="../../clotho_v2/cls1/run_aqa_yn.sh|--hugging_face_model_name_or_path bert-base-uncased"
recipe_runners["aqa_open_bert"]="../../clotho_v2/cls1/run_aqa_open.sh|--hugging_face_model_name_or_path bert-base-uncased"

# CLAP audio-text classification
recipe_runners["cle_clap"]="../../clotho_v2/cls1/run_entailment.sh|--hugging_face_model_name_or_path laion/clap-htsat-unfused"
recipe_runners["aqa_yn_clap"]="../../clotho_v2/cls1/run_aqa_yn.sh|--hugging_face_model_name_or_path laion/clap-htsat-unfused"
recipe_runners["aqa_open_clap"]="../../clotho_v2/cls1/run_aqa_open.sh|--hugging_face_model_name_or_path laion/clap-htsat-unfused"

# General sound Multi-label tasks
recipe_runners["audioset2m"]="../../as2m/cls1/run.sh|"
recipe_runners["audioset20k"]="../../as20k/cls1/run.sh|"
recipe_runners["fsd50k"]="../../fsd50k/cls1/run.sh|"

# ESC-50 folds
recipe_runners["esc50_f1"]="../../esc50/cls1/run_single_fold.sh|--local_data_opts 1 --train_set train1 --valid_set val1 --test_sets val1"
recipe_runners["esc50_f2"]="../../esc50/cls1/run_single_fold.sh|--local_data_opts 2 --train_set train2 --valid_set val2 --test_sets val2"
recipe_runners["esc50_f3"]="../../esc50/cls1/run_single_fold.sh|--local_data_opts 3 --train_set train3 --valid_set val3 --test_sets val3"
recipe_runners["esc50_f4"]="../../esc50/cls1/run_single_fold.sh|--local_data_opts 4 --train_set train4 --valid_set val4 --test_sets val4"
recipe_runners["esc50_f5"]="../../esc50/cls1/run_single_fold.sh|--local_data_opts 5 --train_set train5 --valid_set val5 --test_sets val5"

# Register BEANS detection tasks
recipe_runners["beans_dcase"]="../../beans/cls1/run_dcase.sh|"
recipe_runners["beans_enabirds"]="../../beans/cls1/run_enabirds.sh|"
recipe_runners["beans_gibbons"]="../../beans/cls1/run_gibbons.sh|"
recipe_runners["beans_hiceas"]="../../beans/cls1/run_hiceas.sh|"
recipe_runners["beans_rfcx"]="../../beans/cls1/run_rfcx.sh|"

# Register BEANS classification tasks
recipe_runners["beans_watkins"]="../../beans/cls1/run_watkins.sh|"
recipe_runners["beans_bats"]="../../beans/cls1/run_bats.sh|"
recipe_runners["beans_cbi"]="../../beans/cls1/run_cbi.sh|"
recipe_runners["beans_humbugdb"]="../../beans/cls1/run_humbugdb.sh|"
recipe_runners["beans_dogs"]="../../beans/cls1/run_dogs.sh|"

# Music tasks
recipe_runners["gtzan"]="../../gtzan/cls1/run.sh|"
recipe_runners["nsynth_instrument"]="../../nsynth/cls1/run_instrument.sh|"
recipe_runners["nsynth_pitch"]="../../nsynth/cls1/run_pitch.sh|"

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
                exit 1
            fi
        done
    fi

    if [ ${#recipes_to_run[@]} -eq 0 ]; then
        log "No recipes found matching criteria"
        exit 1
    fi

    echo "${recipes_to_run[@]}"
    return 0
}

create_config() {
    local recipe=$1
    shift 1

    local template_dir="conf/template"
    local config_dir="exp/${run_name}/conf"
    local template_path="${template_dir}/${config_prefix}/${recipe%.yaml}.yaml"
    local output_path="${config_dir}/${config_prefix}/${recipe%.yaml}.yaml"

    mkdir -p "$(dirname "$output_path")"
    if [ ! -f "$template_path" ]; then
        log "Template not found: $template_path"
        exit 1
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
    local config_dir="$(pwd)/exp/${run_name}/conf"
    local config_path="${config_dir}/${config_prefix}/${recipe%.yaml}.yaml"
    local recipe_dir="$(dirname $runner)"

    if [ ! -d "$recipe_dir" ]; then
        log "Error: Recipe directory not found: $recipe_dir"
        exit 1
    fi

    task_name=$(basename $recipe_dir)
    if [[ $task_name == "cls1" ]]; then
        task_name="cls"
    elif [[ $task_name == "asr1" ]]; then
        task_name="asr"
    else
        log "Error: Unknown task: $task_name"
        exit 1
    fi
    cmd_args="--${task_name}_config ${config_path} --${task_name}_tag ${run_name} "
    if "${store_locally}"; then
        cmd_args+="--dumpdir $(pwd)/dump/${recipe} --expdir $(pwd)/exp/${recipe} "
        if [[ $task_name == "cls" ]]; then
            cmd_args+="--datadir $(pwd)/data/${recipe} "
        fi
        # data for asr1 is downloaded in the recipe dir
    fi

    task_args_header="--${task_name}_args"
    task_args_="--use_wandb ${log_wandb} --wandb_project audioverse --wandb_entity shikhar --wandb_name ${recipe}.${run_name}"
    task_args_+=" ${task_args}"
    if [[ "${log_wandb}" == "true" || -n "${task_args}" ]]; then
        cmd_args+="${task_args_header} \"${task_args_}\" "
    fi

    echo "${cmd_args}"
    return 0
}

run_recipe() {
    local recipe=$1
    local runner_info="${recipe_runners[$recipe]}"
    shift

    local success=true
    local recipe_log="${exp_dir}/${recipe}.log"

    log "Processing recipe: $recipe"
    IFS=',' read -ra runner_info_list <<< "$runner_info"

    for runner_entry in "${runner_info_list[@]}"; do
        # Split the runner entry into path and args
        IFS='|' read -ra runner_parts <<< "$runner_entry"
        local runner="${runner_parts[0]}"
        local runner_specific_args="${runner_parts[1]:-}"  # Use empty string if no args specified

        log "Setting up task: $runner for recipe: $recipe with runner args: $runner_specific_args"

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

        log "Running command: $runner with args: ${runner_specific_args} ${cmd_args} ${recipe_args}"
        log "Output will be saved to: $recipe_log"

        command="cd \"$(dirname "$runner")\" && \"./$(basename "$runner")\" ${runner_specific_args} ${cmd_args} ${recipe_args}"

        if [[ "$dry_run" = true ]]; then
            log "Dry run: Would execute: eval \"$command\""
        else
            if ${parallel}; then
                eval "($command 2>&1 | tee -a \"$recipe_log\") &"
                log "Started $recipe|$runner in background (PID: $!)"
                sleep "$wait_time"
            else
                eval "($command 2>&1 | tee -a \"$recipe_log\")" || {
                    log "Error: Recipe $recipe|$runner failed"
                    success=false
                }
            fi
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
    log "------------------------------------------------------------------------------------------------------------------------"
    if run_recipe "$recipe" "$run_args"; then
        successful=$((successful + 1))
        echo "$recipe: SUCCESS" >> "$summary_file"
    else
        failed=$((failed + 1))
        echo "$recipe: FAILED" >> "$summary_file"
    fi
done

log "------------------------------------------------------------------------------------------------------------------------"

# If running in parallel, wait for all background jobs
if ${parallel}; then
    log "Waiting for all recipes to complete..."
    wait

    # For parallel mode, we need to check logs for success/failure
    total=0
    successful=0
    failed=0

    for recipe in $recipes; do
        total=$((total + 1))
        recipe_log="${exp_dir}/${recipe}.log"
        if grep -q "Error\|Failed\|fatal" "$recipe_log"; then
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
