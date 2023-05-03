from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


############################################
# Definitions of shared symbols and values #
############################################
SYMBOL_NA: str = '<na>'             # symbol denoting text is not available
SYMBOL_NOSPEECH: str = '<nospeech>' # symbol denoting non-speech audio
SPEECH_MAX_LEN: float = 30          # max speech length in seconds
SPEECH_RESOLUTION: float = 0.02     # resolution in seconds
# all timestamp symbols
SYMBOLS_TIME: List[str] = [
    "<notimestamps>",
    *[
        f"<{i * SPEECH_RESOLUTION:.2f}>"
        for i in range(round(SPEECH_MAX_LEN / SPEECH_RESOLUTION) + 1)
    ],
]


@dataclass
class Utterance:
    utt_id: str
    wav_id: str
    wav_path: str
    start_time: float   # in seconds
    end_time: float     # in seconds
    lang: str           # language token of speech
    task: str           # task token
    text: str           # target text without timestamps
    asr_text: str       # source text for CTC ASR without timestamps

@dataclass
class LongUtterance(Utterance):
    prev_text: str      # previous (target) text as condition
    text_with_time: str # target text with timestamps


def time2token(x: float) -> str:
    """"Convert float time to timestamp token."""
    x = round(x / SPEECH_RESOLUTION) * SPEECH_RESOLUTION
    return f"<{x:.2f}>"


def merge_short_utterances(utts: List[Utterance], prev: Optional[LongUtterance] = None) -> LongUtterance:
    """Merge a list of utterances to create a long utterance."""

    wav_id = utts[0].wav_id
    wav_path = utts[0].wav_path
    start_time = utts[0].start_time
    end_time = utts[-1].end_time
    lang = utts[0].lang
    task = utts[0].task
    utt_id = f"{wav_id}_{round(1000 * start_time):09d}_{round(1000 * end_time):09d}_{lang[1:-1]}_{task[1:-1]}"
    text = ' '.join([u.text for u in utts])
    asr_text = ' '.join([u.asr_text for u in utts])
    prev_text = prev.text if prev is not None else SYMBOL_NA

    text_with_time = ''
    for u in utts:
        text_with_time += f"{time2token(u.start_time - start_time)} {u.text.strip()}{time2token(u.end_time - start_time)}"
    
    return LongUtterance(
        utt_id=utt_id,
        wav_id=wav_id,
        wav_path=wav_path,
        start_time=start_time,
        end_time=end_time,
        lang=lang,
        task=task,
        text=text,
        asr_text=asr_text,
        prev_text=prev_text,
        text_with_time=text_with_time,
    )


def generate_long_utterances(
    utts: List[Utterance],      # list of short utterances in the same long talk/speech
) -> List[LongUtterance]:
    """"Generate a list of long utterances from a list of short utterances."""

    long_utts = [None]
    l, r = 0, 0
    while l < len(utts):
        if r < len(utts) and utts[r].end_time - utts[l].start_time <= SPEECH_MAX_LEN:
            r += 1
        elif r > l:
            long_utts.append(
                merge_short_utterances(
                    utts[l:r],
                    long_utts[-1]
                )
            )
            l = r
        else:   # skip the current utt if its length already exceeds the limit
            long_utts.append(None)
            l = r + 1
            r = l

    long_utts = [u for u in long_utts if u is not None]
    return long_utts
