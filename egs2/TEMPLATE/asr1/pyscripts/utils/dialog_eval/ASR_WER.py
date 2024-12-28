from typing import Tuple

import numpy as np

from espnet2.sds.utils.utils import int2float


def handle_espnet_ASR_WER(
    ASR_audio_output: Tuple[int, np.ndarray], ASR_transcript: str
) -> str:
    """
    Compute and return Word Error Rate (WER) and Character Error Rate (CER) metrics
    for multiple judge ASR systems (ESPnet, OWSM, Whisper) using the Versa library.

    This function performs the following:
        1. Imports necessary metrics and setup functions from Versa.
        2. Prepares configuration arguments for each ASR system (ESPnet, OWSM, Whisper).
        3. Runs the Levenshtein-based WER/CER calculations.
        4. Returns a formatted string summarizing WER and CER
        results for reference produced by each ASR system.

    Args:
        ASR_audio_output (tuple):
            A tuple where:
                - The first element is the frame rate.
                - The second element is the audio signal (NumPy array).
        ASR_transcript (str):
            The transcript produced by the ASR model in the cascaded
            conversational AI pipeline.

    Returns:
        str:
            A formatted string showing the WER and CER percentages
            for ESPnet, OWSM, and Whisper. Example output:

            "ESPnet WER: 10.50
             ESPnet CER: 7.20
             OWSM WER: 11.30
             OWSM CER: 8.00
             Whisper WER: 9.25
             Whisper CER: 6.50"

    Raises:
        ImportError:
            If Versa is not installed or cannot be imported.

    Example:
        >>> asr_audio_output = (16000, audio_array)
        >>> asr_transcript = "This is the ASR transcript."
        >>> result = handle_espnet_ASR_WER(asr_audio_output, asr_transcript)
        >>> print(result)
        "ESPnet WER: 10.50
         ESPnet CER: 7.20
         OWSM WER: 11.30
         OWSM CER: 8.00
         Whisper WER: 9.25
         Whisper CER: 6.50"
    """
    try:
        from versa import (
            espnet_levenshtein_metric,
            espnet_wer_setup,
            owsm_levenshtein_metric,
            owsm_wer_setup,
            whisper_levenshtein_metric,
            whisper_wer_setup,
        )
    except Exception as e:
        print("Error: Versa is not properly installed.")
        raise e
    score_modules_espnet = {
        "module": espnet_levenshtein_metric,
        "args": espnet_wer_setup(
            model_tag="default",
            beam_size=1,
            text_cleaner="whisper_en",
            use_gpu=True,
        ),
    }
    dict1 = score_modules_espnet["module"](
        score_modules_espnet["args"],
        int2float(ASR_audio_output[1]),
        ASR_transcript,
        ASR_audio_output[0],
    )
    espnet_wer = (
        dict1["espnet_wer_delete"]
        + dict1["espnet_wer_insert"]
        + dict1["espnet_wer_replace"]
    ) / (
        dict1["espnet_wer_insert"]
        + dict1["espnet_wer_replace"]
        + dict1["espnet_wer_equal"]
    )
    espnet_cer = (
        dict1["espnet_cer_delete"]
        + dict1["espnet_cer_insert"]
        + dict1["espnet_cer_replace"]
    ) / (
        dict1["espnet_cer_insert"]
        + dict1["espnet_cer_replace"]
        + dict1["espnet_cer_equal"]
    )
    score_modules_owsm = {
        "module": owsm_levenshtein_metric,
        "args": owsm_wer_setup(
            model_tag="default",
            beam_size=1,
            text_cleaner="whisper_en",
            use_gpu=True,
        ),
    }
    dict1 = score_modules_owsm["module"](
        score_modules_owsm["args"],
        int2float(ASR_audio_output[1]),
        ASR_transcript,
        ASR_audio_output[0],
    )
    owsm_wer = (
        dict1["owsm_wer_delete"] + dict1["owsm_wer_insert"] + dict1["owsm_wer_replace"]
    ) / (dict1["owsm_wer_insert"] + dict1["owsm_wer_replace"] + dict1["owsm_wer_equal"])
    owsm_cer = (
        dict1["owsm_cer_delete"] + dict1["owsm_cer_insert"] + dict1["owsm_cer_replace"]
    ) / (dict1["owsm_cer_insert"] + dict1["owsm_cer_replace"] + dict1["owsm_cer_equal"])
    score_modules_whisper = {
        "module": whisper_levenshtein_metric,
        "args": whisper_wer_setup(
            model_tag="default",
            beam_size=1,
            text_cleaner="whisper_en",
            use_gpu=True,
        ),
    }
    dict1 = score_modules_whisper["module"](
        score_modules_whisper["args"],
        int2float(ASR_audio_output[1]),
        ASR_transcript,
        ASR_audio_output[0],
    )
    whisper_wer = (
        dict1["whisper_wer_delete"]
        + dict1["whisper_wer_insert"]
        + dict1["whisper_wer_replace"]
    ) / (
        dict1["whisper_wer_insert"]
        + dict1["whisper_wer_replace"]
        + dict1["whisper_wer_equal"]
    )
    whisper_cer = (
        dict1["whisper_cer_delete"]
        + dict1["whisper_cer_insert"]
        + dict1["whisper_cer_replace"]
    ) / (
        dict1["whisper_cer_insert"]
        + dict1["whisper_cer_replace"]
        + dict1["whisper_cer_equal"]
    )
    return (
        f"ESPnet WER: {espnet_wer*100:.2f}\n"
        f"ESPnet CER: {espnet_cer*100:.2f}\n"
        f"OWSM WER: {owsm_wer*100:.2f}\n"
        f"OWSM CER: {owsm_cer*100:.2f}\n"
        f"Whisper WER: {whisper_wer*100:.2f}\n"
        f"Whisper CER: {whisper_cer*100:.2f}"
    )
