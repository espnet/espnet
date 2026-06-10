from typing import Tuple

import numpy as np

from espnet2.sds.utils.utils import int2float


def TTS_psuedomos(TTS_audio_output: Tuple[int, np.ndarray]) -> str:
    """
    Compute and return speech quality metrics
    for the given synthesized audio output
    using the Versa library.

    Args:
        TTS_audio_output (Tuple[int, np.ndarray]):
            A tuple containing:
                - The first element (int): The frame rate of the audio.
                - The second element (np.ndarray): The audio signal,
                typically a NumPy array.

    Returns:
        str:
            A formatted string containing each metric name
            and its corresponding score, for example:

            utmos: 3.54
            dnsmos: 3.47
            plcmos: 3.62
            sheet_ssqa: 4.03

    Raises:
        ImportError:
            If the Versa library is not installed or cannot be imported.

    Example:
        >>> tts_audio_output = (16000, audio_array)
        >>> result = TTS_psuedomos(tts_audio_output)
        >>> print(result)
        utmos: 3.54
        dnsmos: 3.47
        plcmos: 3.62
        sheet_ssqa: 4.03
    """
    try:
        from versa import (
            pseudo_mos_metric,
            pseudo_mos_setup,
            sheet_ssqa,
            sheet_ssqa_setup,
        )
    except Exception as e:
        print("Error: Versa is not properly installed.")
        raise e

    predictor_dict, predictor_fs = pseudo_mos_setup(
        use_gpu=True,
        predictor_types=["utmos", "dnsmos", "plcmos"],
        predictor_args={
            "utmos": {"fs": 16000},
            "dnsmos": {"fs": 16000},
            "plcmos": {"fs": 16000},
        },
    )
    score_modules = {
        "module": pseudo_mos_metric,
        "args": {
            "predictor_dict": predictor_dict,
            "predictor_fs": predictor_fs,
            "use_gpu": True,
        },
    }
    dict1 = score_modules["module"](
        int2float(TTS_audio_output[1]),
        TTS_audio_output[0],
        **score_modules["args"],
    )
    str1 = ""
    for k in dict1:
        str1 = str1 + f"{k}: {dict1[k]:.2f}\n"
    sheet_model = sheet_ssqa_setup(
        model_tag="default",
        model_path=None,
        model_config=None,
        use_gpu=True,
    )
    score_modules = {
        "module": sheet_ssqa,
        "args": {"model": sheet_model, "use_gpu": True},
    }
    dict1 = score_modules["module"](
        score_modules["args"]["model"],
        int2float(TTS_audio_output[1]),
        TTS_audio_output[0],
        use_gpu=score_modules["args"]["use_gpu"],
    )
    for k in dict1:
        str1 = str1 + f"{k}: {dict1[k]:.2f}\n"
    return str1
