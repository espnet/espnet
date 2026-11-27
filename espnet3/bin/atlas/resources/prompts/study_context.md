Read study context with tools as needed; do not ask for pasted copies.
- Objective: ${objective}
- Knowledge pack: ${knowledge_pack}
- Trial history: ${trial_history}
- Resolved Atlas config: ${atlas_config}
- Graph: ${graph}
- Recipe directory: ${recipe_dir}
For every propose request, use Read in this agent call on Objective (program.md), Resolved Atlas config, Trial history, and Knowledge pack before reasoning. Read current trial status and scores from those files; do not describe a pending job as already training. These files are mutable and supersede any resumed-session/cache memory. Then read the recipe config paths named by atlas.yaml before proposing or modifying a trial. Do not propose a patch outside the currently read search_space.

Recipe-isolation rule: this is an independent experiment. Read and use only the listed study paths, files inside the current recipe directory, and ESPnet framework code only when needed to understand an API. Do not list, search, read, compare against, or reuse information from any other recipe under egs3/, including sibling or parent-reachable recipe paths. If you accidentally access another recipe, do not use that information; report the scope violation and return a failure rather than proposing, editing, or accepting an experiment.
