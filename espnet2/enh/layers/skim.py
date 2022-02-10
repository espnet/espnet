# An implementation of SkiM model described in 
# "SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech Separation"
# (https://arxiv.org/abs/2201.10800)
# 


from turtle import forward, shape
import torch
import torch.nn as nn

from espnet2.enh.layers.dprnn import SingleRNN, split_feature, merge_feature
from espnet2.enh.layers.tcn import chose_norm


class MemLSTM(nn.Module):
    """ the Mem-LSTM of SkiM

    args:
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional. Default is False.
        mem_type: 'hc', 'h', 'c' or 'id'. 
            It controls whether the hidden (or cell) state of SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will be identically returned.  
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(self,hidden_size, dropout=0.0, bidirectional=False, mem_type='hc', norm_type='gLN'):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type

        assert mem_type in ["hc", "h", 'c', 'id'], f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"

        if mem_type in ["hc", 'h']:
            self.h_net = SingleRNN('LSTM', input_size=self.input_size, hidden_size=self.hidden_size, dropout=dropout, bidirectional=bidirectional)
            self.h_norm = chose_norm(norm_type=norm_type, channel_size=self.input_size, shape='BTD')
        if mem_type in ["hc", 'c']:
            self.c_net = SingleRNN('LSTM', input_size=self.input_size, hidden_size=self.hidden_size, dropout=dropout, bidirectional=bidirectional)
            self.c_norm = chose_norm(norm_type=norm_type, channel_size=self.input_size, shape='BTD')
    
    def extra_repr(self) -> str:
        return f"Mem_type: {self.mem_type}, bidirectional: {self.bidirectional}"

    def forward(self, hc, S):
        # hc = (h, c), tuple of hidden and cell states from SegLSTM
        # shape of h and c: (d, B*S, H)
        # S: number of segments in SegLSTM
        
        if self.mem_type == 'id':
            ret_val = hc
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH  
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH  
            if self.mem_type == 'hc':
                h = h + self.h_norm(self.h_net(h))
                c = c + self.c_norm(self.c_net(c))
            elif self.mem_type == 'h':
                h = h + self.h_norm(self.h_net(h))
                c = torch.zeros_like(c)
            elif self.mem_type == 'c':
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c))
            
            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)

        if not self.bidirectional:
            # for causal setup
            causal_ret_val = []
            for x in ret_val:
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1,:]
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)
        
        return ret_val
   
            


class SegLSTM(nn.Module):

    """ the Seg-LSTM of SkiM

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).  
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional. Default is False.
    """

    def __init__(self, input_size, hidden_size, dropout=0.0, bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.lstm = nn.LSTM(input_size, hidden_size, 1,batch_first=True,bidirectional=bidirectional,)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)  
    
    def forward(self, input, hc):
        # input shape: B, T, H

        B, T, H = input.shape

        if hc == None:
            # In fist input SkiM block, h and c are not available
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size).to(input.device)
            c = torch.zeros(d, B, self.hidden_size).to(input.device)
        else:
            h, c = hc

        output, (h, c) = self.lstm(input, (h, c)) 
        output = self.dropout(output)
        output = self.proj(
            output.contiguous().view(-1, output.shape[2])
        ).view(output.shape)       

        return output, (h, c)


