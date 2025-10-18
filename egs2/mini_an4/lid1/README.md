# Mini AN4 LID Recipe

## Overview

This recipe provides a minimal Language Identification (LID) setup using the AN4 dataset for **CI testing purposes only**.

## Important Notes

**This recipe is designed specifically for CI validation, not for real LID training.**

- **Test Case Purpose**: The mini_an4 recipe serves as a test case for CI validation to ensure LID functionality works correctly.

- **Speaker ID as Language ID**: Since mini_an4 does not contain actual language labels, we use speaker IDs as language IDs for testing purposes. This allows the LID pipeline to run and be validated without requiring multilingual data.

- **CI Testing Only**: This setup is intended to verify that all LID components (data preparation, training, inference) function correctly in the CI environment.

## Usage

```bash
# Run the complete pipeline for CI testing
./run.sh --stage 1 --stop_stage 8
```

## For Real LID Training

For actual Language Identification tasks, please refer to other LID recipes like `egs2/voxlingua107/lid1` that use genuine multilingual datasets with proper language labels.
