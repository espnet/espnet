# Copyright 2024
# Licensed under the MIT License
# For details, see: https://github.com/gpt-omni/mini-omni?tab=MIT-1-ov-file#readme

# This code is borrowed from the following sources:
# 1. https://github.com/gpt-omni/mini-omni
# 2. https://huggingface.co/spaces/gradio/omni-mini/tree/main

import time

import numpy as np
import torch


class SnacConfig:
    audio_vocab_size = 4096
    padded_vocab_size = 4160
    end_of_audio = 4097


snac_config = SnacConfig()


def get_time_str():
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return time_str


def layershift(input_id, layer, stride=4160, shift=152000):
    return input_id + shift + layer * stride


def generate_audio_data(snac_tokens, snacmodel, device=None):
    audio = reconstruct_tensors(snac_tokens, device)
    with torch.inference_mode():
        audio = [torch.clamp(k, max=4095) for k in audio]
        audio_hat = snacmodel.decode(audio)
    audio_data = audio_hat.cpu().numpy().astype(np.float64) * 32768.0
    audio_data = audio_data.astype(np.int16)
    audio_data = audio_data.tobytes()
    return audio_data


def get_snac(list_output, index, nums_generate):

    snac = []
    start = index
    for i in range(nums_generate):
        snac.append("#")
        for j in range(7):
            snac.append(list_output[j][start - nums_generate - 5 + j + i])
    return snac


def reconscruct_snac(output_list):
    if len(output_list) == 8:
        output_list = output_list[:-1]
    output = []
    for i in range(7):
        output_list[i] = output_list[i][i + 1 :]
    for i in range(len(output_list[-1])):
        output.append("#")
        for j in range(7):
            output.append(output_list[j][i])
    return output


def reconstruct_tensors(flattened_output, device=None):
    """Reconstructs the list of tensors from the flattened output."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def count_elements_between_hashes(lst):
        try:
            # Find the index of the first '#'
            first_index = lst.index("#")
            # Find the index of the second '#' after the first
            second_index = lst.index("#", first_index + 1)
            # Count the elements between the two indices
            return second_index - first_index - 1
        except ValueError:
            # Handle the case where there aren't enough '#' symbols
            return "List does not contain two '#' symbols"

    def remove_elements_before_hash(flattened_list):
        try:
            # Find the index of the first '#'
            first_hash_index = flattened_list.index("#")
            # Return the list starting from the first '#'
            return flattened_list[first_hash_index:]
        except ValueError:
            # Handle the case where there is no '#'
            return "List does not contain the symbol '#'"

    def list_to_torch_tensor(tensor1):
        # Convert the list to a torch tensor
        tensor = torch.tensor(tensor1)
        # Reshape the tensor to have size (1, n)
        tensor = tensor.unsqueeze(0)
        return tensor

    flattened_output = remove_elements_before_hash(flattened_output)
    codes = []
    tensor1 = []
    tensor2 = []
    tensor3 = []
    tensor4 = []

    n_tensors = count_elements_between_hashes(flattened_output)
    if n_tensors == 7:
        for i in range(0, len(flattened_output), 8):

            tensor1.append(flattened_output[i + 1])
            tensor2.append(flattened_output[i + 2])
            tensor3.append(flattened_output[i + 3])
            tensor3.append(flattened_output[i + 4])

            tensor2.append(flattened_output[i + 5])
            tensor3.append(flattened_output[i + 6])
            tensor3.append(flattened_output[i + 7])
            codes = [
                list_to_torch_tensor(tensor1).to(device),
                list_to_torch_tensor(tensor2).to(device),
                list_to_torch_tensor(tensor3).to(device),
            ]

    if n_tensors == 15:
        for i in range(0, len(flattened_output), 16):

            tensor1.append(flattened_output[i + 1])
            tensor2.append(flattened_output[i + 2])
            tensor3.append(flattened_output[i + 3])
            tensor4.append(flattened_output[i + 4])
            tensor4.append(flattened_output[i + 5])
            tensor3.append(flattened_output[i + 6])
            tensor4.append(flattened_output[i + 7])
            tensor4.append(flattened_output[i + 8])

            tensor2.append(flattened_output[i + 9])
            tensor3.append(flattened_output[i + 10])
            tensor4.append(flattened_output[i + 11])
            tensor4.append(flattened_output[i + 12])
            tensor3.append(flattened_output[i + 13])
            tensor4.append(flattened_output[i + 14])
            tensor4.append(flattened_output[i + 15])

            codes = [
                list_to_torch_tensor(tensor1).to(device),
                list_to_torch_tensor(tensor2).to(device),
                list_to_torch_tensor(tensor3).to(device),
                list_to_torch_tensor(tensor4).to(device),
            ]

    return codes
