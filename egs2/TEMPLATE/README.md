# TEMPLATE
## How to make new recipe? 

1. Create directory in egs/
    ```bash
    % task=asr1  # tts1, mt1, st1
    % egs/TEMPLATE/${task}/setup.sh egs/foo/${task}
    ```
   
1. Create run.sh and local/ somehow
    ```bash
    % cd egs/foo/${task}
    % cp ../../mini_an4/${task}/run.sh .
    % vi run.sh
    ```
   
1. If the recipe uses some corpora and they are not listed in `db.sh`, then write it.
1. If the recipe depends on some special tools, then write the requirements to `path.sh`

    path.sh:
    ```bash
    # e.g. flac command is required
    if ! which flac &> /dev/null; then 
        echo "Error: flac is not installed"
        return 1
    fi
    ```
