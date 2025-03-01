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
filter_recipe="" # Filter recipes by name. Default none
recipe="" # Specific recipes to run, takes precedence. Default is all
run_name=$(date "+%Y%m%d%H%M%S") # Uniqueness
log_dir="logs/${run_name}"

# Help message
help_message=$(cat << EOF
Usage: $0 [OPTIONS] [ARGS_FOR_RECIPES...]
Run multiple recipes under Audioverse.

Options:
  --parallel BOOL           Run recipes in parallel (default: ${parallel})
  --wait_time SECONDS       Wait time between parallel runs (default: ${wait_time})
  --filter_recipe STRING    Run all recipes except these. Default is None.
  --recipe LIST             Comma-separated list of specific recipes to run. Default is all.
  --run_name NAME           Name for this benchmark run (default: timestamp)

    We process two types of arguments for recipes:
    1. Recipe-specific arguments: These replace ALL_UPPER_CASE strings in conf/template/recipe_name_train.conf
    For example BEATS_CHECKPOINT_PATH in conf/template/clotho_v2_asr1_train.conf will be replaced by the value of flag --BEATS_CHECKPOINT_PATH provided by you.
    2. Recipe-common arguments: These are passed to all recipes and override conf files.
EOF
)


. utils/parse_options.sh
. ./path.sh


mkdir -p "$log_dir"
log "Created log directory: $log_dir"

# Register all task related variables that are unique
declare -A task_unique_args
task_unique_args["cls1"]="cls_config,cls_tag"
task_unique_args["asr1"]="asr_config,asr_tag"

# Register all recipe runners
declare -A recipe_runners
recipe_runners["beans"]="beans/cls1/run.sh"

# recipe_runners["clotho_aqa"]="clotho_v2/cls1/run_aqa_yn.sh,clotho_v2/cls1/run_aqa_open.sh"
# recipe_runners["clotho_cle"]="clotho_v2/cls1/run_entailment.sh"
# recipe_runners["clotho_aac"]="clotho_v2/asr1/run.sh"

# recipe_runners["audioset2m"]="as2m/cls1/run.sh"
# recipe_runners["audioset20k"]="as20k/cls1/run.sh"


create_config() {
    local recipe=$1
    local task=$2
    shift 2  # Remove the first two arguments
    
    local template_dir="conf/template"
    local config_dir="conf/${run_name}"
    local template_path="${template_dir}/${recipe}/${task}/train.conf"
    local output_path="${config_dir}/${recipe}/${task}/train.conf"
    
    # Create directory for the config
    mkdir -p "$(dirname "$output_path")"
    
    # Check if template exists
    if [ ! -f "$template_path" ]; then
        log "Warning: Template not found: $template_path"
        return 1
    fi
    
    # Copy template and replace ARGUMENT_NAME placeholders
    cp "$template_path" "$output_path"
    sed -i "s|BEATS_CHECKPOINT_PATH|$checkpoint_path|g" "$output_path"
    
    # Replace other arguments from command line
    for arg in "$@"; do
        if [[ $arg == --*=* ]]; then
            arg_name="${arg#--}"
            arg_name="${arg_name%%=*}"
            arg_name=$(echo "$arg_name" | tr '[:lower:]' '[:upper:]')
            arg_value="${arg#*=}"
            sed -i "s|${arg_name}|$arg_value|g" "$output_path"
        fi
    done
    
    log "Created config file: $output_path"
    return 0
}


generate_recipe_list() {
    local recipes_to_run=()
    
    # Add all recipes except filtered ones
    for r in "${!recipe_runners[@]}"; do
        [[ -n "$filter_recipe" && $r != *"$filter_recipe"* ]] && recipes_to_run+=("$r")
    done
    
    if [ -n "$recipe" ]; then
        recipes_to_run=() # Reset
        IFS=',' read -ra requested_recipes <<< "$recipe"
        for r in "${requested_recipes[@]}"; do
            if [ -n "${recipe_runners[$r]}" ]; then
                recipes_to_run+=("$r")
            else
                log "Warning: Unknown recipe: $r"
            fi
        done
    fi
    
    if [ ${#recipes_to_run[@]} -eq 0 ]; then
        log "No recipes found matching criteria"
        return 1
    fi
    
    log "Selected recipes: ${recipes_to_run[*]}"
    echo "${recipes_to_run[@]}"
    return 0
}

# Function to construct command for a recipe
construct_command() {
    local recipe=$1
    local task=$2
    local config_dir="conf/${run_name}"
    local recipe_dir="egs2/${recipe}/${task}"
    local config_path="${config_dir}/${recipe}/${task}/train.conf"
    
    # Check if recipe directory exists
    if [ ! -d "$recipe_dir" ]; then
        log "Error: Recipe directory not found: $recipe_dir"
        return 1
    fi
    
    # Build command with appropriate config parameter
    local cmd="${recipe_dir}/run.sh"
    
    # Add task-specific arguments
    IFS=',' read -ra task_vars <<< "${task_unique_args[$task]}"
    for var in "${task_vars[@]}"; do
        if [[ $var == *_config ]]; then
            cmd="$cmd --${var}=${config_path}"
        elif [[ $var == *_tag ]]; then
            cmd="$cmd --${var}=${run_name}"
        fi
    done
    
    echo "$cmd"
    return 0
}


run_recipe() {
    local recipe=$1
    local tasks="${recipe_runners[$recipe]}"
    shift
    
    local success=true
    local recipe_log="${log_dir}/${recipe}.log"
    
    log "Processing recipe: $recipe"
    IFS=',' read -ra task_list <<< "$tasks"
    
    for task in "${task_list[@]}"; do
        log "Setting up task: $task for recipe: $recipe"
        
        # Create config
        create_config "$recipe" "$task" "$@" || {
            log "Error: Failed to create config for $recipe/$task"
            success=false
            continue
        }
        
        # Construct command
        local cmd=$(construct_command "$recipe" "$task") || {
            log "Error: Failed to construct command for $recipe/$task"
            success=false
            continue
        }
        
        # Add remaining arguments (pass through all remaining args)
        for arg in "$run_args"; do
            # Filter out args that were already handled by parse_options.sh
            cmd="$cmd $arg"
        done
        
        log "Running command: $cmd"
        log "Output will be saved to: $recipe_log"
        
        if [ "$parallel" = true ]; then
            # Run in background if parallel
            (cd "$(dirname "$cmd")" && "./$(basename "$cmd")" 2>&1 | tee -a "$recipe_log") &
            log "Started $recipe/$task in background (PID: $!)"
            sleep "$wait_time"
        else
            # Run and wait for completion
            (cd "$(dirname "$cmd")" && "./$(basename "$cmd")" 2>&1 | tee -a "$recipe_log") || {
                log "Error: Recipe $recipe/$task failed"
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


summary_file="${log_dir}/summary.txt"
echo "Benchmark Run: $run_name" > "$summary_file"
echo "Checkpoint: $checkpoint_path" >> "$summary_file"
echo "Started: $(date)" >> "$summary_file"
echo "----------------------------------------" >> "$summary_file"

recipes=$(generate_recipe_list) || {
    log "Error: Failed to generate recipe list"
    exit 1
}

total=0
successful=0
failed=0

# Run them all
for recipe in $recipes; do
    total=$((total + 1))
    if run_recipe "$recipe" "$@"; then
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
        recipe_log="${log_dir}/${recipe}.log"
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