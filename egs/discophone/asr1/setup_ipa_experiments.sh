#!/bin/bash

python3 local/prepare_experiment_configs.py
for config in conf/experiments/*.conf; do
  expname="$(basename "${config//.conf/}")"
  echo Creating $expname from $config
  ./setup_experiment.sh "$expname"
  sed -i "s:langs_config=:langs_config=${config}:" "../$expname/run.sh"
done

