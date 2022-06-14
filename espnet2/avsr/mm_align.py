import torch
import math
from typeguard import check_argument_types

def truncate_align(rate_ratio:float, speech:torch.Tensor, speech_lengths: torch.Tensor,
                    vision:torch.Tensor, vision_lengths:torch.Tensor) -> torch.Tensor:
    """
    Truncate the high sample ratio source to match the low sample ratio source
    """
    assert check_argument_types()
    assert speech.size(0) == vision.size(0), (speech.shape, vision.shape)
    assert speech.dim() == 3, speech.shape
    assert vision.dim() == 3, vision.shape

    device = speech.device
    source_unchanged = vision if rate_ratio > 1 else speech
    source_to_modify = speech if rate_ratio > 1 else vision
    feat_lengths = vision_lengths if rate_ratio > 1 else speech_lengths
    ratio = rate_ratio if rate_ratio > 1 else 1 / rate_ratio

    B, _, F = source_to_modify.size()
    _, T, _ = source_unchanged.size()

    batch_indices = []
    for b in range(B):
        indices = torch.round(torch.arange(0, len(source_unchanged[b])) * ratio).long()
        indices = torch.clamp(indices, min = 0, max = len(source_to_modify[b]) - 1).unsqueeze(-1)
        indices = indices.repeat(1, F).unsqueeze(0) 
        batch_indices.append(indices)

    indices = torch.cat(batch_indices)
    assert indices.size() == (B, T, F), indices.shape
    indices = indices.to(device)
    source_to_modify = torch.gather(source_to_modify, 1, indices)
    assert source_to_modify.size() == (B, T, F), source_to_modify.shape
    speech = source_to_modify if rate_ratio > 1 else source_unchanged
    vision = source_unchanged if rate_ratio > 1 else source_to_modify
    assert(speech.size(0) == vision.size(0),
           speech.size(1) == vision.size(1)
        ), (speech.shape, vision.shape)
    return speech, vision, feat_lengths

def avg_pool_align(rate_ratio:float, speech:torch.Tensor, speech_lengths: torch.Tensor, 
                    vision:torch.Tensor, vision_lengths:torch.Tensor) -> torch.Tensor:
    """
    Average Pool the high sample ratio source to match the low sample ratio source
    """
    assert check_argument_types()
    assert speech.size(0) == vision.size(0), (speech.shape, vision.shape)
    assert speech.dim() == 3, speech.shape
    assert vision.dim() == 3, vision.shape
    
    device = speech.device
    source_unchanged = vision if rate_ratio > 1 else speech
    source_to_modify = speech if rate_ratio > 1 else vision
    feat_lengths = vision_lengths if rate_ratio > 1 else speech_lengths
    ratio = rate_ratio if rate_ratio > 1 else 1 / rate_ratio
    
    B, _, F = source_to_modify.size()
    _, T, _ = source_unchanged.size()

    indices = torch.round(torch.arange(0, source_unchanged.size(1)) * ratio).long()
    indices = torch.clamp(indices, min = 0, max = source_to_modify.size(1) - 1)

    prev_index = 0
    batch_pool = []
    for b in range(B):
        avg_pool = []
        for index in indices:
            if index == 0:
                avg_pool.append(source_to_modify[b,index,:].unsqueeze(0))
            else:
                avg_pool.append(torch.mean(source_to_modify[b,prev_index:index,:], 0).unsqueeze(0))
            prev_index = index
        batch_pool.append(torch.cat(avg_pool).unsqueeze(0))
    
    source_to_modify = torch.cat(batch_pool).to(device)
    assert source_to_modify.size() == (B, T, F), source_to_modify.shape
    speech = source_to_modify if rate_ratio > 1 else source_unchanged
    vision = source_unchanged if rate_ratio > 1 else source_to_modify
    assert(speech.size(0) == vision.size(0),
           speech.size(1) == vision.size(1)
        ), (speech.shape, vision.shape)
    return speech, vision, feat_lengths

def duplicate_align(rate_ratio:float, speech:torch.Tensor, speech_lengths: torch.Tensor, 
                    vision:torch.Tensor, vision_lengths:torch.Tensor) -> torch.Tensor:
    """
    Duplicate the low sample ratio source to match the high sample ratio source
    """
    assert check_argument_types()
    assert speech.size(0) == vision.size(0), (speech.shape, vision.shape)
    assert speech.dim() == 3, speech.shape
    assert vision.dim() == 3, vision.shape

    device = speech.device
    source_unchanged = speech if rate_ratio > 1 else vision
    source_to_modify = vision if rate_ratio > 1 else speech
    feat_lengths = speech_lengths if rate_ratio > 1 else vision_lengths
    ratio = 1 / rate_ratio if rate_ratio > 1 else rate_ratio

    B, _, F = source_to_modify.size()
    _, T, _ = source_unchanged.size()

    batch_indices = []
    for b in range(B):
        indices = torch.round(torch.arange(0, len(source_unchanged[b])) * ratio).long()
        indices = torch.clamp(indices, min = 0, max = len(source_to_modify[b]) - 1).unsqueeze(-1)
        indices = indices.repeat(1, F).unsqueeze(0) 
        batch_indices.append(indices)
    
    indices = torch.cat(batch_indices)
    assert indices.size() == (B, T, F), indices.shape
    indices = indices.to(device)
    source_to_modify = torch.gather(source_to_modify, 1, indices)
    assert source_to_modify.size() == (B, T, F), source_to_modify.shape
    speech = source_unchanged if rate_ratio > 1 else source_to_modify
    vision = source_to_modify if rate_ratio > 1 else source_unchanged
    assert(speech.size(0) == vision.size(0),
           speech.size(1) == vision.size(1)
        ), (speech.shape, vision.shape)
    return speech, vision, feat_lengths


ALIGN_OPTIONS = {"truncate": truncate_align,
                 "avg_pool": avg_pool_align,
                 "duplicate": duplicate_align}

class MM_Aligner():
    def __init__(self, align_option):
        if align_option not in ALIGN_OPTIONS.keys():
            raise ValueError("align_option should be one of {}".format(ALIGN_OPTIONS))
        self.align_func = ALIGN_OPTIONS[align_option]

    def align(self, speech:torch.Tensor, speech_lengths:torch.Tensor, 
                vision:torch.Tensor, vision_lengths:torch.Tensor, ssr = None, vsr = None):
        assert speech_lengths.dim() == 1, speech_lengths.shape
        assert vision_lengths.dim() == 1, vision_lengths.shape

        if ssr is not None and vsr is not None:
            # Work with Rate Numbers
            rate_ratio = ssr / vsr
            if ssr == vsr:
                length = min(vision.size(1), speech.size(1))
                speech = speech[:,:length,:]
                vision = vision[:,:length,:]
                feat_lengths = torch.min(
                    torch.cat([speech_lengths, vision_lengths], dim = 1), dim = 1)[0]
            else:
                speech, vision, feat_lengths = self.align_func(
                    rate_ratio, speech, speech_lengths, vision, vision_lengths)
        else:
            # Work with input lengths
            rate_ratio = float(torch.mean(speech_lengths / vision_lengths).item())
            speech, vision, feat_lengths = self.align_func(
                rate_ratio, speech, speech_lengths, vision, vision_lengths)
        assert(speech.size(0) == vision.size(0),
               speech.size(1) == vision.size(1),
              ), (speech.shape, vision.shape)
        assert (speech.size(0) == feat_lengths.size(0),
                feat_lengths.dim() == 1,
               ), feat_lengths.shape
        return speech, vision, feat_lengths