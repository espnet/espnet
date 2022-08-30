from contextlib import contextmanager
from distutils.version import LooseVersion
from collections import defaultdict
import argparse
import torch
import sys
import numpy as np
from tqdm import tqdm

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

from espnet2.tasks.st import STTask
from espnet2.tasks.mt import MTTask
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.torch_utils.device_funcs import to_device
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.st.espnet_model_hier import ESPnetSTHierModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

def compute_gcd(x, y):

   while(y):
       x, y = y, x % y
   return x

def compute_lcm(x, y):
   lcm = (x*y)//compute_gcd(x,y)
   return lcm

def diagonality(att):
    # att (torch.Tensor): (B, NHeads, Tout, Tin)

    if isinstance(att, torch.Tensor):
        att = att.detach().cpu().numpy()
    bs, nheads, len_out, len_in = att.shape
    result = np.zeros([bs, nheads])
    for b in range(bs):
        for n in range(nheads):
            max_input = np.argmax(att[b][n], axis=-1)
            monotonic = 0
            for i in range(len(max_input)):
                if i == 0 :
                    monotonic += 1
                elif max_input[i] >= max_input[i-1]:
                    monotonic += 1
            result[b][n] = monotonic/len(max_input)
    # (B, NHeads)
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

    log_fp = open(args.log_file, 'w')
    #log_fp.write(' '.join(sys.argv) + '\n')

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
    #outputs = {}
    #handles = {}
    #for name, modu in mt_model.named_modules():
    #    if "mt_lego_encoder" not in name and "target_encoder" in name and "src_attn" in name and "extra_asr_decoder" not in name:
    #        def hook(module, input, output, name=name):
    #            if isinstance(module, MultiHeadedAttention):
    #                # NOTE(kamo): MultiHeadedAttention doesn't return attention weight
    #                # attn: (B, Head, Tout, Tin)
    #                outputs[name] = module.attn.detach().cpu()

    #        handle = modu.register_forward_hook(hook)
    #        handles[name] = handle

    # iterate over all samples
    return_dict = dict()

    for keys, batch in tqdm(loader):
        # assert isinstance(batch, dict), type(batch)
        # assert all(isinstance(s, str) for s in keys), keys
        # _bs = len(next(iter(batch.values())))
        # assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

        src_text = batch["src_text"].to(torch.long)  # (B, T)
        src_text_lengths = batch["src_text_lengths"].to(torch.long)

        text = batch["text"].to(torch.long)
        text_lengths = batch["text_lengths"].to(torch.long)

        speech = batch["speech"].to(getattr(torch, "float32"))  # (B, T)
        lengths = batch["speech_lengths"]

        batch = {"src_text": src_text, "src_text_lengths": src_text_lengths, "text": text, "text_lengths": text_lengths, "speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=args.device)

        # b. Forward Encoder
        loss, stats, weight = st_model(**batch)
        return_dict[keys[0]] = (stats['loss_st_att'].item() , stats['loss_mt_ctc'].item())
        #if isinstance(mt_model, ESPnetMTModelDCTC):
        #    enc, enc_lens, _ = mt_model.mt_lego_encoder(enc, enc_lens)

        ## Forward Decoder
        #ys_in_pad, ys_out_pad = add_sos_eos(text, mt_model.sos, mt_model.eos, mt_model.ignore_id)
        #ys_in_lens = text_lengths + 1
        #dec, _ = mt_model.decoder(
        #        enc, enc_lens, ys_in_pad, ys_in_lens
        #    )

        # Derive the attention results

    for name, diags in return_dict.items():
        # diags = np.concatenate(diags, axis=0)     # (num_samples, num_heads)
        loss = diags[0] + diags[1]
        log_fp.write(
            f"{name} {loss} {diags[0]} {diags[1]}\n"
        )
    
    log_fp.close()
