#!/usr/bin/env bats

setup() {
    tmpdir=/tmp/espnet-test-plot-sinc-filters-${RANDOM}
    mkdir ${tmpdir}
    export LC_ALL="en_US.UTF-8"

    # Create dummy model
    python << EOF
import matplotlib
import os
import torch

from espnet2.layers.sinc_conv import SincConv

output_path="${tmpdir}"
model_path = output_path + "/test.model.pth"
# We need a mock model. - One could also initialize a full E2E model.
filters = SincConv(
    in_channels=1, out_channels=128, kernel_size=101, stride=1, fs=16000
)
model = {"preencoder.filters.f": filters.f}
model = {"model": model}
torch.save(model, model_path)

EOF
}

@test "plot_sinc_filters.py" {
    if ! which plot_sinc_filters.py &> /dev/null; then
        skip
    fi

    python plot_sinc_filters.py --filetype svg --sample_rate 16000 "${tmpdir}/test.model.pth" ${tmpdir}
}

teardown() {
    rm -r $tmpdir
}
