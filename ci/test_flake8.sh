#!/usr/bin/env bash

. tools/activate_python.sh

set -euo pipefail

# TODO: remove files from this list!
flake8_black_list="\
espnet/asr/asr_utils.py
espnet/asr/chainer_backend/asr.py
espnet/asr/pytorch_backend/asr.py
espnet/bin/asr_enhance.py
espnet/lm/chainer_backend/extlm.py
espnet/lm/chainer_backend/lm.py
espnet/lm/lm_utils.py
espnet/lm/pytorch_backend/extlm.py
espnet/nets/chainer_backend/ctc.py
espnet/nets/chainer_backend/deterministic_embed_id.py
espnet/nets/chainer_backend/nets_utils.py
espnet/nets/chainer_backend/rnn/attentions.py
espnet/nets/chainer_backend/rnn/decoders.py
espnet/nets/chainer_backend/rnn/encoders.py
espnet/nets/chainer_backend/rnn/training.py
espnet/nets/chainer_backend/transformer/mask.py
espnet/nets/ctc_prefix_score.py
espnet/nets/pytorch_backend/ctc.py
espnet/nets/pytorch_backend/frontends/beamformer.py
espnet/nets/pytorch_backend/frontends/dnn_beamformer.py
espnet/nets/pytorch_backend/frontends/dnn_wpe.py
espnet/nets/pytorch_backend/frontends/feature_transform.py
espnet/nets/pytorch_backend/frontends/frontend.py
espnet/nets/pytorch_backend/frontends/mask_estimator.py
espnet/nets/pytorch_backend/nets_utils.py
espnet/nets/pytorch_backend/rnn/attentions.py
espnet/nets/pytorch_backend/rnn/decoders.py
espnet/nets/pytorch_backend/rnn/encoders.py
espnet/nets/pytorch_backend/streaming/segment.py
espnet/nets/pytorch_backend/streaming/window.py
espnet/nets/pytorch_backend/transformer/plot.py
espnet/nets/pytorch_backend/wavenet.py
espnet/transform/add_deltas.py
espnet/transform/channel_selector.py
espnet/transform/cmvn.py
espnet/transform/functional.py
espnet/transform/perturb.py
espnet/transform/spec_augment.py
espnet/transform/spectrogram.py
espnet/transform/transform_interface.py
espnet/transform/transformation.py
espnet/transform/wpe.py
espnet/utils/check_kwargs.py
espnet/utils/cli_readers.py
espnet/utils/cli_utils.py
espnet/utils/cli_writers.py
espnet/utils/deterministic_utils.py
espnet/utils/dynamic_import.py
espnet/utils/fill_missing_args.py
espnet/utils/io_utils.py
espnet/utils/spec_augment.py
espnet/utils/training/batchfy.py
espnet/utils/training/evaluator.py
espnet/utils/training/iterators.py
espnet/utils/training/tensorboard_logger.py
espnet/utils/training/train_utils.py
"

n_blacklist=$(wc -l <<< "${flake8_black_list}")
n_all=$(find espnet -name "*.py" | wc -l)
n_ok=$((n_all - n_blacklist))
cov=$(echo "scale = 4; 100 * ${n_ok} / ${n_all}" | bc)
echo "flake8-docstrings ready files coverage: ${n_ok} / ${n_all} = ${cov}%"

# --extend-ignore for wip files for flake8-docstrings
flake8 --show-source --extend-ignore=D test utils doc ${flake8_black_list} espnet2 test/espnet2 egs/*/*/local/*.py

# white list of files that should support flake8-docstrings
flake8 --show-source espnet --exclude=${flake8_black_list//$'\n'/,}
