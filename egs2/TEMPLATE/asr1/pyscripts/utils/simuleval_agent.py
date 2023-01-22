# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction

from espnet2.bin.st_inference import Speech2Text
from espnet2.bin.st_inference_streaming import Speech2TextStreaming
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from typing import Any, List, Optional, Sequence, Tuple, Union
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
import torch
import logging


@entrypoint
class DummyAgent(SpeechToTextAgent):
    """
    DummyAgent operates in an offline mode.
    Waits until all source is read to run inference.
    """

    def __init__(self, args):
        super().__init__(args)
        kwargs = vars(args)
    
        logging.basicConfig(
            level=kwargs['log_level'],
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

        if kwargs['ngpu'] >= 1:
            device = "cuda"
        else:
            device = "cpu"

        # 1. Set random-seed
        set_all_random_seed(kwargs['seed'])

        # 2. Build speech2text
        if kwargs['backend'] == 'offline':
            speech2text_kwargs = dict(
                st_train_config=kwargs['st_train_config'],
                st_model_file=kwargs['st_model_file'],
                transducer_conf=kwargs['transducer_conf'],
                lm_train_config=kwargs['lm_train_config'],
                lm_file=kwargs['lm_file'],
                ngram_file=kwargs['ngram_file'],
                src_lm_train_config=kwargs['src_lm_train_config'],
                src_lm_file=kwargs['src_lm_file'],
                src_ngram_file=kwargs['src_ngram_file'],
                token_type=kwargs['token_type'],
                bpemodel=kwargs['bpemodel'],
                src_token_type=kwargs['src_token_type'],
                src_bpemodel=kwargs['src_bpemodel'],
                device=device,
                maxlenratio=kwargs['maxlenratio'],
                minlenratio=kwargs['minlenratio'],
                asr_maxlenratio=kwargs['asr_maxlenratio'],
                asr_minlenratio=kwargs['asr_minlenratio'],
                dtype=kwargs['dtype'],
                beam_size=kwargs['beam_size'],
                ctc_weight=kwargs['ctc_weight'],
                lm_weight=kwargs['lm_weight'],
                ngram_weight=kwargs['ngram_weight'],
                penalty=kwargs['penalty'],
                nbest=kwargs['nbest'],
                asr_beam_size=kwargs['asr_beam_size'],
                asr_ctc_weight=kwargs['asr_ctc_weight'],
                asr_lm_weight=kwargs['asr_lm_weight'],
                asr_ngram_weight=kwargs['asr_ngram_weight'],
                asr_penalty=kwargs['asr_penalty'],
                asr_nbest=kwargs['asr_nbest'],
                enh_s2t_task=kwargs['enh_s2t_task'],
                ctc_greedy=kwargs['ctc_greedy']
            )
            self.speech2text = Speech2Text.from_pretrained(
                model_tag=kwargs['model_tag'],
                **speech2text_kwargs,
            )
        else:
            speech2text_kwargs = dict(
                st_train_config=kwargs['st_train_config'],
                st_model_file=kwargs['st_model_file'],
                lm_train_config=kwargs['lm_train_config'],
                lm_file=kwargs['lm_file'],
                token_type=kwargs['token_type'],
                bpemodel=kwargs['bpemodel'],
                device=device,
                maxlenratio=kwargs['maxlenratio'],
                minlenratio=kwargs['minlenratio'],
                dtype=kwargs['dtype'],
                beam_size=kwargs['beam_size'],
                ctc_weight=kwargs['ctc_weight'],
                lm_weight=kwargs['lm_weight'],
                penalty=kwargs['penalty'],
                nbest=kwargs['nbest'],
                disable_repetition_detection=kwargs['disable_repetition_detection'],
                decoder_text_length_limit=kwargs['decoder_text_length_limit'],
                encoded_feat_length_limit=kwargs['encoded_feat_length_limit'],
            )
            self.speech2text = Speech2TextStreaming(**speech2text_kwargs)
        
        self.sim_chunk_length = kwargs['sim_chunk_length']
        self.backend = kwargs['backend']
        self.clean()

    @staticmethod
    def add_args(parser):
        # Note(kamo): Use '_' instead of '-' as separator.
        # '-' is confusing if written in yaml.
        parser.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )

        parser.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--dtype",
            default="float32",
            choices=["float16", "float32", "float64"],
            help="Data type",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )

        group = parser.add_argument_group("The model configuration related")
        group.add_argument(
            "--st_train_config",
            type=str,
            help="ST training configuration",
        )
        group.add_argument(
            "--st_model_file",
            type=str,
            help="ST model parameter file",
        )
        group.add_argument(
            "--lm_train_config",
            type=str,
            help="LM training configuration",
        )
        group.add_argument(
            "--src_lm_train_config",
            type=str,
            help="LM training configuration",
        )
        group.add_argument(
            "--lm_file",
            type=str,
            help="LM parameter file",
        )
        group.add_argument(
            "--src_lm_file",
            type=str,
            help="LM parameter file",
        )
        group.add_argument(
            "--word_lm_train_config",
            type=str,
            help="Word LM training configuration",
        )
        group.add_argument(
            "--src_word_lm_train_config",
            type=str,
            help="Word LM training configuration",
        )
        group.add_argument(
            "--word_lm_file",
            type=str,
            help="Word LM parameter file",
        )
        group.add_argument(
            "--src_word_lm_file",
            type=str,
            help="Word LM parameter file",
        )
        group.add_argument(
            "--ngram_file",
            type=str,
            help="N-gram parameter file",
        )
        group.add_argument(
            "--src_ngram_file",
            type=str,
            help="N-gram parameter file",
        )
        group.add_argument(
            "--model_tag",
            type=str,
            help="Pretrained model tag. If specify this option, *_train_config and "
            "*_file will be overwritten",
        )
        group.add_argument(
            "--enh_s2t_task",
            type=str2bool,
            default=False,
            help="enhancement and asr joint model",
        )

        group = parser.add_argument_group("Beam-search related")
        group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="The batch size for inference",
        )
        group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
        group.add_argument("--asr_nbest", type=int, default=1, help="Output N-best hypotheses")
        group.add_argument("--beam_size", type=int, default=20, help="Beam size")
        group.add_argument("--asr_beam_size", type=int, default=20, help="Beam size")
        group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
        group.add_argument("--asr_penalty", type=float, default=0.0, help="Insertion penalty")
        group.add_argument(
            "--maxlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain max output length. "
            "If maxlenratio=0.0 (default), it uses a end-detect "
            "function "
            "to automatically find maximum hypothesis lengths."
            "If maxlenratio<0.0, its absolute value is interpreted"
            "as a constant max output length",
        )
        group.add_argument(
            "--asr_maxlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain max output length. "
            "If maxlenratio=0.0 (default), it uses a end-detect "
            "function "
            "to automatically find maximum hypothesis lengths."
            "If maxlenratio<0.0, its absolute value is interpreted"
            "as a constant max output length",
        )
        group.add_argument(
            "--minlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain min output length",
        )
        group.add_argument(
            "--asr_minlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain min output length",
        )
        group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
        group.add_argument("--asr_lm_weight", type=float, default=1.0, help="RNNLM weight")
        group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
        group.add_argument("--asr_ngram_weight", type=float, default=0.9, help="ngram weight")
        group.add_argument("--ctc_weight", type=float, default=0.0, help="ST CTC weight")
        group.add_argument("--asr_ctc_weight", type=float, default=0.3, help="ASR CTC weight")

        group.add_argument(
            "--transducer_conf",
            default=None,
            help="The keyword arguments for transducer beam search.",
        )

        group = parser.add_argument_group("Text converter related")
        group.add_argument(
            "--token_type",
            type=str_or_none,
            default=None,
            choices=["char", "bpe", None],
            help="The token type for ST model. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--src_token_type",
            type=str_or_none,
            default=None,
            choices=["char", "bpe", None],
            help="The token type for ST model. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model path of sentencepiece. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--src_bpemodel",
            type=str_or_none,
            default=None,
            help="The model path of sentencepiece. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--ctc_greedy",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            "--sim_chunk_length",
            type=int,
            default=0,
            help="The length of one chunk, to which speech will be "
            "divided for evalution of streaming processing.",
        )
        group.add_argument("--disable_repetition_detection", type=str2bool, default=False)
        group.add_argument(
            "--encoded_feat_length_limit",
            type=int,
            default=0,
            help="Limit the lengths of the encoded feature" "to input to the decoder.",
        )
        group.add_argument(
            "--decoder_text_length_limit",
            type=int,
            default=0,
            help="Limit the lengths of the text" "to input to the decoder.",
        )

        group.add_argument(
            "--backend",
            type=str,
            default="offline",
            help="Limit the lengths of the text" "to input to the decoder.",
        )

        return parser

    def clean(self):
        self.processed_index = -1
        self.maxlen = 0

    def policy(self):

        # dummy offline policy
        if self.backend == 'offline':
            if self.states.source_finished:
                results = self.speech2text(torch.tensor(self.states.source))
                if self.speech2text.st_model.use_multidecoder:
                    prediction = results[0][0][0] # multidecoder result is in this format
                else:
                    prediction = results[0][0]
                return WriteAction(prediction, finished=True)
            else:
                return ReadAction()
        
        # hacked streaming policy. takes running beam search hyp as an incremental output 
        else:
            unread_length = len(self.states.source) - self.processed_index - 1
            if unread_length >= self.sim_chunk_length or self.states.source_finished:
                speech = torch.tensor(self.states.source[self.processed_index+1:])
                results = self.speech2text(speech=speech, is_final=self.states.source_finished)
                self.processed_index = len(self.states.source) - 1
                if not self.states.source_finished:
                    if self.speech2text.beam_search.running_hyps and len(self.speech2text.beam_search.running_hyps.yseq[0]) > self.maxlen:
                        prediction = self.speech2text.beam_search.running_hyps.yseq[0][1:]
                        prediction = self.speech2text.converter.ids2tokens(prediction)
                        prediction = self.speech2text.tokenizer.tokens2text(prediction)
                        self.maxlen = len(self.speech2text.beam_search.running_hyps.yseq[0])
                    else:
                        return ReadAction()
                else:
                    prediction = results[0][0]

                unwritten_length = len(prediction) - len("".join(self.states.target))
                # if unwritten_length > 0:
                print(self.processed_index, prediction[-unwritten_length:])
                if self.states.source_finished:
                    self.clean()
                return WriteAction(prediction[-unwritten_length:], finished=self.states.source_finished)

            return ReadAction()