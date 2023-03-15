from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.rnn.encoders import RNN

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class RNNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "blstm",
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "sigmoid",
        layer: int = 3,
        unit: int = 512,
        dropout: float = 0.0,
    ):
        """RNN Separator

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'blstm', 'lstm' etc.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            dropout: float, dropout ratio. Default is 0.
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.rnn = RNN(
            idim=input_dim,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(unit, input_dim) for _ in range(num_outputs)]
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        x, ilens, _ = self.rnn(feature, ilens)

        masks = []

        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
    
    def forward_streaming(self, input_frame: torch.Tensor, states=None):
        # input_frame # B, 1, N

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input_frame)
        else:
            feature = input_frame

        ilens = torch.ones(feature.shape[0], device=feature.device)
        
        x, _, states = self.rnn(feature, ilens, states)

        masks = []

        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input_frame * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise
        
        return masked, states, others
    

if __name__ == "__main__":

    import humanfriendly
    import time

    torch.set_num_threads(1)

    SEQ_LEN = 10000
    num_spk = 2
    separator = RNNSeparator(
        input_dim = 128,
        rnn_type="lstm",
        num_spk=2,
        layer=7,
        unit=400,
    )
    separator.eval()

    print(f"Number of parameters: {humanfriendly.format_size(sum(p.numel() for p in separator.parameters()))}" )

    input_feature = torch.randn((16, SEQ_LEN, 128))
    ilens = torch.LongTensor([SEQ_LEN] * 16)

    with torch.no_grad():
        
        start = time.time()
        seq_output, _, _ = separator.forward(input_feature, ilens=ilens)
        end = time.time()
        print(f"seqeunce processing, time cost: {end - start}")

        start = time.time()
        state = None
        stream_outputs = []
        for i in range(SEQ_LEN):
            frame = input_feature[:,i:i+1,:]
            frame_out, state, _ = separator.forward_streaming(frame, state)
            stream_outputs.append(frame_out)
        end = time.time()
        print(f"streaming processing, time cost: {end - start}")         
        for i in range(SEQ_LEN):
            for s in range(num_spk):
                torch.testing.assert_allclose(stream_outputs[i][s], seq_output[s][:, i:i+1, :])
    

    print("Streaming OKey")