from contextlib import contextmanager
from distutils.version import LooseVersion
from collections import defaultdict
import argparse
import torch
import sys
import numpy as np
from tqdm import tqdm

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.attention import RelPositionMultiHeadedAttention


from espnet2.tasks.st import STTask
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.torch_utils.device_funcs import to_device
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos



if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def diagonality(att):
    # att (torch.Tensor): (B, NHeads, Tout, Tin)
    if isinstance(att, torch.Tensor):
        att = att.detach().cpu().numpy()
    bs, nheads, len_out, len_in = att.shape

    rel_distance = np.zeros((len_out, len_in))
    for i in range(len_out):
        for j in range(len_in):
            rel_distance[i, j] = np.abs(i - j)
    
    result = (1. - (att * rel_distance).sum(-1) / rel_distance.max(-1)).mean(-1)    # (B, NHeads)
    return result


def format_array(arr):
    string = np.array2string(
        arr,
        formatter={'float_kind': lambda x: f"{x:.6f}"},
        separator=','
    )
    return string


def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate the diagonality of the self-attention weights."
    )
    parser.add_argument(
        "--st_train_config",
        type=str,
        help="path to the asr train config file"
    )
    parser.add_argument(
        "--st_model_file",
        type=str,
        help="path to the trained model file"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help="device name: cpu (default), gpu"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="log file name"
    )
    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    parser.add_argument(
        "--key_file", 
        type=str_or_none,
        help="wav.scp"
    )
    parser.add_argument(
        "--allow_variable_data_keys", 
        type=str2bool, 
        default=False
    )
    # parser.add_argument(
    #     "--use_hier_ctc", 
    #     type=str2bool, 
    #     default=True
    # )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    log_fp = open(args.log_file, 'a')
    log_fp.write(' '.join(sys.argv) + '\n')

    if args.device == 'gpu':
        args.device = 'cuda'
    
    st_model, st_train_args = STTask.build_model_from_file(
        args.st_train_config, args.st_model_file, args.device
    )
    st_model.eval()

    # dataloader
    loader = STTask.build_streaming_iterator(
        args.data_path_and_name_and_type,
        dtype="float32",
        batch_size=1,       # mush be 1, otherwise there will be paddings
        key_file=args.key_file,
        num_workers=2,
        preprocess_fn=STTask.build_preprocess_fn(st_train_args, False),
        collate_fn=STTask.build_collate_fn(st_train_args, False),
        allow_variable_data_keys=args.allow_variable_data_keys,
        inference=False,
    )

    # forward hook
    outputs = {}
    handles = {}
    for name, modu in st_model.named_modules():
        if "encoder" in name and "self_attn" in name:
            def hook(module, input, output, name=name):
                if isinstance(module, MultiHeadedAttention) or isinstance(module, RelPositionMultiHeadedAttention):
                    # NOTE(kamo): MultiHeadedAttention doesn't return attention weight
                    # attn: (B, Head, Tout, Tin)
                    outputs[name] = module.attn.detach().cpu()

            handle = modu.register_forward_hook(hook)
            handles[name] = handle
    
    # iterate over all samples
    return_dict = defaultdict(list)

    for keys, batch in tqdm(loader):
        # assert isinstance(batch, dict), type(batch)
        # assert all(isinstance(s, str) for s in keys), keys
        # _bs = len(next(iter(batch.values())))
        # assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

        speech = batch["speech"].to(getattr(torch, "float32"))  # (B, T)
        lengths = batch["speech_lengths"]
        text = batch["text"]
        text_lengths = batch["text_lengths"]
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=args.device)

        # b. Forward Encoder
        enc, enc_lens = st_model.encode(**batch)

        enc_hier, enc_hier_lens, _ = st_model.encoder_hier(enc, enc_lens)

        # # Forward Decoder
        # ys_in_pad, ys_out_pad = add_sos_eos(text, st_model.sos, st_model.eos, st_model.ignore_id)
        # ys_in_lens = text_lengths + 1
        # dec, _ = st_model.decoder(
        #         enc_hier, enc_hier_lens, ys_in_pad, ys_in_lens
        #     )

        # Derive the attention results
        for name, output in outputs.items():
            # name: e.g., encoder.encoders.23.self_attn
            # output: (Batch, NHead, Tout, Tin)
            diag = diagonality(output)      # (B, NHeads)
            return_dict[name].append(diag)

        outputs.clear()

    # 3. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return_dict = dict(return_dict)

    for name, diags in return_dict.items():
        diags = np.concatenate(diags, axis=0)     # (num_samples, num_heads)
        log_fp.write(
            f"{name}: nsamples={diags.shape[0]}, mean={format_array(diags.mean(0))}, std={format_array(diags.std(0))}\n"
        )
    
    log_fp.close()