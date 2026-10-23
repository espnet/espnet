#!/usr/bin/env bash

recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${recipe_dir}${PYTHONPATH:+:${PYTHONPATH}}"
