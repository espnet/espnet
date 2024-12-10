# An implementation of SkiM model described in
# "SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech Separation"
# (https://arxiv.org/abs/2201.10800)
#

import torch
import torch.nn as nn

from espnet2.enh.layers.dprnn import SingleRNN, merge_feature, split_feature
from espnet2.enh.layers.tcn import choose_norm


class MemLSTM(nn.Module):
    """
    Memory LSTM (MemLSTM) for the SkiM model.

    This class implements the MemLSTM layer used in the SkiM model described in
    "SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech 
    Separation" (https://arxiv.org/abs/2201.10800).

    Attributes:
        hidden_size (int): Dimension of the hidden state.
        dropout (float): Dropout ratio. Default is 0.
        bidirectional (bool): Whether the LSTM layers are bidirectional.
            Default is False.
        mem_type (str): Controls how the hidden (or cell) state of SegLSTM
            will be processed by MemLSTM. Options are 'hc', 'h', 'c', or 'id'.
        norm_type (str): Normalization type. Options are 'gLN' or 'cLN'.
            'cLN' is for causal implementation.

    Args:
        hidden_size (int): Dimension of the hidden state.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the LSTM layers are 
            bidirectional. Default is False.
        mem_type (str, optional): Controls how the hidden (or cell) state 
            of SegLSTM will be processed by MemLSTM. Options are 'hc', 
            'h', 'c', or 'id'. Default is 'hc'.
        norm_type (str, optional): Normalization type. Options are 'gLN' 
            or 'cLN'. Default is 'cLN'.

    Raises:
        AssertionError: If `mem_type` is not one of the supported types 
            ('hc', 'h', 'c', 'id').

    Examples:
        >>> mem_lstm = MemLSTM(hidden_size=128, dropout=0.1, 
        ...                     bidirectional=True, mem_type='hc', 
        ...                     norm_type='gLN')
        >>> hc = (torch.randn(2, 32, 128), torch.randn(2, 32, 128))  # (h, c)
        >>> output = mem_lstm(hc, S=4)  # S is the number of segments

    Note:
        The forward method expects `hc` to be a tuple of hidden and cell
        states, and `S` to represent the number of segments.
    """

    def __init__(
        self,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        mem_type="hc",
        norm_type="cLN",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
        ], f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"

        if mem_type in ["hc", "h"]:
            self.h_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.h_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )
        if mem_type in ["hc", "c"]:
            self.c_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.c_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )

    def extra_repr(self) -> str:
        """
        Returns a string representation of the MemLSTM module's configuration.

    This method provides information about the memory type and whether the 
    LSTM layers are bidirectional, which is useful for debugging and logging 
    purposes.

    Attributes:
        mem_type (str): The type of memory used by the MemLSTM. Can be one of 
            'hc', 'h', 'c', or 'id'.
        bidirectional (bool): Indicates if the LSTM layers are bidirectional.

    Returns:
        str: A string summarizing the configuration of the MemLSTM.

    Examples:
        >>> mem_lstm = MemLSTM(hidden_size=128, bidirectional=True, mem_type='hc')
        >>> print(mem_lstm.extra_repr())
        'Mem_type: hc, bidirectional: True'
        """
        return f"Mem_type: {self.mem_type}, bidirectional: {self.bidirectional}"

    def forward(self, hc, S):
        """
        Forward pass for the MemLSTM layer.

    This method processes the hidden and cell states from the SegLSTM and
    applies the MemLSTM transformation based on the specified memory type.
    The function handles both identity mode and different memory types ('hc',
    'h', 'c') to compute the new hidden and cell states.

    Args:
        hc (tuple): A tuple containing the hidden and cell states from SegLSTM.
            Each state should have the shape (d, B*S, H), where:
            - d: number of directions (1 for unidirectional, 2 for bidirectional)
            - B: batch size
            - S: number of segments
            - H: hidden size
        S (int): Number of segments in the SegLSTM.

    Returns:
        tuple: A tuple containing the updated hidden and cell states.
            The shape will be (B*S, d, H) for each state.

    Note:
        If `self.mem_type` is set to 'id', the function returns the input
        hidden and cell states unchanged. If `self.bidirectional` is False,
        the output will be modified for causal processing.

    Examples:
        >>> mem_lstm = MemLSTM(hidden_size=128, mem_type='hc')
        >>> h = torch.randn(2, 10, 128)  # Example hidden state
        >>> c = torch.randn(2, 10, 128)  # Example cell state
        >>> hc = (h, c)
        >>> S = 5  # Example number of segments
        >>> output_hc = mem_lstm.forward(hc, S)
        """
        # hc = (h, c), tuple of hidden and cell states from SegLSTM
        # shape of h and c: (d, B*S, H)
        # S: number of segments in SegLSTM

        if self.mem_type == "id":
            ret_val = hc
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            if self.mem_type == "hc":
                h = h + self.h_norm(self.h_net(h)[0])
                c = c + self.c_norm(self.c_net(c)[0])
            elif self.mem_type == "h":
                h = h + self.h_norm(self.h_net(h)[0])
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c)[0])

            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)

        if not self.bidirectional:
            # for causal setup
            causal_ret_val = []
            for x in ret_val:
                x = x.transpose(1, 0).contiguous().view(B, S, d * H)
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                x_ = x_.view(B * S, d, H).transpose(1, 0).contiguous()
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)

        return ret_val

    def forward_one_step(self, hc, state):
        """
        Forward one step in the MemLSTM processing.

        This method computes the next hidden and cell states given the current
        hidden and cell states. It processes the input based on the memory type
        specified during the initialization of the MemLSTM class. 

        Args:
            hc (tuple): A tuple containing the current hidden state (h) and cell 
                state (c). The shapes are expected to be (d, B, H), where d is 
                the number of directions (1 for unidirectional, 2 for 
                bidirectional), B is the batch size, and H is the hidden size.
            state (list): A list containing the hidden states for the LSTM layers, 
                which should match the structure defined during the initialization.

        Returns:
            tuple: A tuple containing the updated hidden and cell states 
                (hc) and the updated state list. The shapes of the hidden and 
                cell states will remain (d, B, H).

        Note:
            This method does not modify the states when mem_type is set to "id".

        Examples:
            >>> mem_lstm = MemLSTM(hidden_size=128, mem_type='hc')
            >>> hc = (torch.zeros(2, 4, 128), torch.zeros(2, 4, 128))  # Example shapes
            >>> state = [torch.zeros(2, 4, 128), torch.zeros(2, 4, 128)]  # Example states
            >>> new_hc, new_state = mem_lstm.forward_one_step(hc, state)

        Raises:
            ValueError: If the mem_type is not one of the supported types.
        """
        if self.mem_type == "id":
            pass
        else:
            h, c = hc
            d, B, H = h.shape
            h = h.transpose(1, 0).contiguous().view(B, 1, d * H)  # B, 1, dH
            c = c.transpose(1, 0).contiguous().view(B, 1, d * H)  # B, 1, dH
            if self.mem_type == "hc":
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            elif self.mem_type == "h":
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            h = h.transpose(1, 0).contiguous()
            c = c.transpose(1, 0).contiguous()
            hc = (h, c)

        return hc, state


class SegLSTM(nn.Module):
    """
    The Seg-LSTM of SkiM.

    This class implements the Segmented Long Short-Term Memory (Seg-LSTM) 
    model as part of the SkiM architecture for low-latency real-time 
    continuous speech separation. It processes input features in segments 
    and maintains hidden states for the LSTM.

    Args:
        input_size (int): Dimension of the input feature. The input 
            should have shape (batch, seq_len, input_size).
        hidden_size (int): Dimension of the hidden state.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the LSTM layers are 
            bidirectional. Default is False.
        norm_type (str, optional): Normalization type, either 'gLN' 
            or 'cLN'. 'cLN' is for causal implementation.

    Returns:
        output (torch.Tensor): The processed output of shape 
            (batch, seq_len, input_size).
        (h, c) (tuple): The hidden and cell states of the LSTM.

    Examples:
        >>> import torch
        >>> model = SegLSTM(input_size=16, hidden_size=32)
        >>> input_tensor = torch.randn(4, 10, 16)  # (batch_size, seq_len, input_size)
        >>> output, (h, c) = model(input_tensor, None)
        >>> print(output.shape)  # Should print: torch.Size([4, 10, 16])

    Note:
        In the first input to the SkiM block, the hidden (h) and cell (c) 
        states are initialized to zero.
    """

    def __init__(
        self, input_size, hidden_size, dropout=0.0, bidirectional=False, norm_type="cLN"
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = choose_norm(
            norm_type=norm_type, channel_size=input_size, shape="BTD"
        )

    def forward(self, input, hc):
        """
        Performs a forward pass of the SegLSTM model.

    This method takes the hidden and cell states from the previous
    SegLSTM layer and processes them through the MemLSTM, applying
    necessary transformations based on the specified memory type.

    Args:
        hc (tuple): A tuple containing hidden and cell states from
            the previous layer. Each of shape (d, B*S, H), where:
            - d: number of directions (1 or 2 for bidirectional)
            - B: batch size
            - S: number of segments
            - H: hidden state size
        S (int): Number of segments in the SegLSTM.

    Returns:
        tuple: A tuple containing the updated hidden and cell states.
            The shapes depend on the `mem_type` configuration and may
            vary as follows:
            - If `mem_type` is 'id', returns the input `hc`.
            - Otherwise, returns transformed hidden and cell states of
              shape (B*S, d, H).

    Raises:
        AssertionError: If the `mem_type` is not one of the allowed values.

    Examples:
        >>> model = SegLSTM(input_size=16, hidden_size=11)
        >>> hc = (torch.zeros(1, 3, 11), torch.zeros(1, 3, 11))  # Example states
        >>> S = 2  # Number of segments
        >>> output = model.forward(hc, S)
    
    Note:
        This method handles both bidirectional and unidirectional
        configurations. If the model is configured for causal
        processing, the output states will be adjusted accordingly.
        """
        # input shape: B, T, H

        B, T, H = input.shape

        if hc is None:
            # In fist input SkiM block, h and c are not available
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
            c = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
        else:
            h, c = hc

        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view(
            input.shape
        )
        output = input + self.norm(output)

        return output, (h, c)


class SkiM(nn.Module):
    """
    Skipping Memory Net (SkiM) for low-latency real-time continuous speech separation.

This class implements the SkiM model as described in the paper:
"SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech 
Separation" (https://arxiv.org/abs/2201.10800).

Attributes:
    input_size (int): Dimension of the input feature.
    hidden_size (int): Dimension of the hidden state.
    output_size (int): Dimension of the output size.
    dropout (float): Dropout ratio. Default is 0.
    num_blocks (int): Number of basic SkiM blocks.
    segment_size (int): Segmentation size for splitting long features.
    bidirectional (bool): Whether the RNN layers are bidirectional.
    mem_type (str or None): Controls whether the hidden (or cell) state 
        of SegLSTM will be processed by MemLSTM. Options are 'hc', 'h', 
        'c', 'id', or None. In 'id' mode, both hidden and cell states 
        will be identically returned. When mem_type is None, the MemLSTM 
        will be removed.
    norm_type (str): Normalization type; can be 'gLN' or 'cLN'. cLN 
        is for causal implementation.
    seg_overlap (bool): Whether the segmentation will reserve 50% overlap 
        for adjacent segments. Default is False.

Args:
    input_size (int): Dimension of the input feature.
    hidden_size (int): Dimension of the hidden state.
    output_size (int): Dimension of the output size.
    dropout (float): Dropout ratio. Default is 0.
    num_blocks (int): Number of basic SkiM blocks.
    segment_size (int): Segmentation size for splitting long features.
    bidirectional (bool): Whether the RNN layers are bidirectional.
    mem_type (str or None): Controls whether the hidden (or cell) state 
        of SegLSTM will be processed by MemLSTM.
    norm_type (str): Normalization type; can be 'gLN' or 'cLN'.
    seg_overlap (bool): Whether to reserve 50% overlap for adjacent segments.

Examples:
    >>> model = SkiM(
    ...     input_size=16,
    ...     hidden_size=11,
    ...     output_size=16,
    ...     dropout=0.0,
    ...     num_blocks=4,
    ...     segment_size=20,
    ...     bidirectional=False,
    ...     mem_type="hc",
    ...     norm_type="cLN",
    ...     seg_overlap=False,
    ... )
    >>> input_tensor = torch.randn(3, 100, 16)
    >>> output = model(input_tensor)

Note:
    This implementation is designed for continuous speech separation 
    tasks with a focus on low-latency processing.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="hc",
        norm_type="gLN",
        seg_overlap=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type
        self.seg_overlap = seg_overlap

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
            None,
        ], f"only support 'hc', 'h', 'c', 'id', and None, current type: {mem_type}"

        self.seg_lstms = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_lstms.append(
                SegLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                )
            )
        if self.mem_type is not None:
            self.mem_lstms = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_lstms.append(
                    MemLSTM(
                        hidden_size,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        mem_type=mem_type,
                        norm_type=norm_type,
                    )
                )
        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )

    def forward(self, input):
        """
        Forward pass of the MemLSTM module.

    This method takes the hidden and cell states from a SegLSTM and processes
    them through the MemLSTM layer, returning the updated hidden and cell 
    states. The method supports various memory types to control how the 
    hidden and cell states are processed.

    Args:
        hc (tuple): A tuple containing the hidden state (h) and cell state (c) 
            from the SegLSTM. Both should have the shape (d, B*S, H), where 
            d is the number of directions, B is the batch size, S is the number 
            of segments, and H is the hidden size.
        S (int): The number of segments in the SegLSTM.

    Returns:
        tuple: A tuple containing the updated hidden state (h) and cell state 
        (c) after processing through the MemLSTM. If the memory type is "id", 
        the original states are returned without modification.

    Note:
        For the causal setup (non-bidirectional), the method adjusts the 
        hidden and cell states to ensure causality.

    Examples:
        >>> mem_lstm = MemLSTM(hidden_size=64, mem_type='hc')
        >>> h = torch.randn(2, 3, 64)  # Example hidden state
        >>> c = torch.randn(2, 3, 64)  # Example cell state
        >>> hc = (h, c)
        >>> updated_hc = mem_lstm.forward(hc, S=3)
        """
        # input shape: B, T (S*K), D
        B, T, D = input.shape

        if self.seg_overlap:
            input, rest = split_feature(
                input.transpose(1, 2), segment_size=self.segment_size
            )  # B, D, K, S
            input = input.permute(0, 3, 2, 1).contiguous()  # B, S, K, D
        else:
            input, rest = self._padfeature(input=input)
            input = input.view(B, -1, self.segment_size, D)  # B, S, K, D
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.view(B * S, K, D).contiguous()  # BS, K, D
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)  # BS, K, D
            if self.mem_type and i < self.num_blocks - 1:
                hc = self.mem_lstms[i](hc, S)
                pass

        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)  # B, D, K, S
            output = merge_feature(output, rest)  # B, D, T
            output = self.output_fc(output).transpose(1, 2)

        else:
            output = output.view(B, S * K, D)[:, :T, :]  # B, T, D
            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size

        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return input, rest

    def forward_stream(self, input_frame, states):
        """
        Process a single frame of input in a streaming manner.

        This method updates the internal state of the SkiM model based on the 
        provided input frame and the current states. It allows the model to 
        handle streaming inputs efficiently by maintaining memory across 
        segments.

        Args:
            input_frame (torch.Tensor): The input frame of shape (B, 1, N), 
                where B is the batch size and N is the feature dimension.
            states (dict): A dictionary containing the current states of the 
                model, including the current step and segment states.

        Returns:
            output (torch.Tensor): The processed output of shape (B, 1, D), 
                where D is the output feature dimension.
            states (dict): The updated states dictionary, which includes the 
                current step and updated segment states.

        Note:
            The `states` dictionary is initialized if not provided. It keeps 
            track of the current step and segment-level states for each 
            block.

        Examples:
            >>> model = SkiM(input_size=16, hidden_size=11, output_size=16)
            >>> input_frame = torch.randn(3, 1, 16)  # Batch of 3, 1 time step
            >>> states = {}
            >>> output, states = model.forward_stream(input_frame, states)

        Todo:
            - Consider optimizing memory usage when dealing with long 
              sequences.
        """
        # input_frame # B, 1, N

        B, _, N = input_frame.shape

        def empty_seg_states():
            shp = (1, B, self.hidden_size)
            return (
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
            )

        B, _, N = input_frame.shape
        if not states:
            states = {
                "current_step": 0,
                "seg_state": [empty_seg_states() for i in range(self.num_blocks)],
                "mem_state": [[None, None] for i in range(self.num_blocks - 1)],
            }

        output = input_frame

        if states["current_step"] and (states["current_step"]) % self.segment_size == 0:
            tmp_states = [empty_seg_states() for i in range(self.num_blocks)]
            for i in range(self.num_blocks - 1):
                tmp_states[i + 1], states["mem_state"][i] = self.mem_lstms[
                    i
                ].forward_one_step(states["seg_state"][i], states["mem_state"][i])

            states["seg_state"] = tmp_states

        for i in range(self.num_blocks):
            output, states["seg_state"][i] = self.seg_lstms[i](
                output, states["seg_state"][i]
            )

        states["current_step"] += 1

        output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output, states


if __name__ == "__main__":
    torch.manual_seed(111)

    seq_len = 100

    model = SkiM(
        16,
        11,
        16,
        dropout=0.0,
        num_blocks=4,
        segment_size=20,
        bidirectional=False,
        mem_type="hc",
        norm_type="cLN",
        seg_overlap=False,
    )
    model.eval()

    input = torch.randn(3, seq_len, 16)
    seg_output = model(input)

    state = None
    for i in range(seq_len):
        input_frame = input[:, i : i + 1, :]
        output, state = model.forward_stream(input_frame=input_frame, states=state)
        torch.testing.assert_allclose(output, seg_output[:, i : i + 1, :])

    print("streaming ok")
