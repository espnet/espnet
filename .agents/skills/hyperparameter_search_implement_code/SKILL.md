---
name: hyperparameter_search_implement_code
description: Implement source changes required by a proposed hyperparameter search trial.
---

# Implement trial code changes

Use the implementation task, trial patch, and allowed directory list supplied in the request.

- Write or modify files only inside the allowed directories relative to the recipe root.
- If the configuration patch is sufficient, make no code changes.
- Follow the recipe's existing style and keep changes limited to this trial.
- For checkpoint loading, always use `load_state_dict(..., strict=True)`; never use `strict=False`.
- Add dependencies only when permitted by the request, using the configured package manager.
- The trial patch, rationale, and `code_edit_config` are supplied as context values. Inspect files only under its `allowed_dirs` and create those directories if needed.
- Report the files actually changed and a short summary in the requested structured output.
- Return `status: success` and a short `message`, along with the requested file/package lists and `summary`.
