#!/usr/bin/env python3
# encoding: utf-8

"""Generate new segments file with WebRTC VAD tool.

A modified version of https://github.com/wiseman/py-webrtcvad/blob/master/example.py.
"""


import argparse
import codecs
import collections
import contextlib
import logging
import os
import wave
import webrtcvad


def get_parser():
    parser = argparse.ArgumentParser(
        description="split data using WebRTC VAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, help="input data dir")
    parser.add_argument(
        "--frame-duration", type=int, default=20, help="milliseconds of each frames"
    )
    parser.add_argument(
        "--padding-duration",
        type=int,
        default=200,
        help="milliseconds padding to each segment",
    )
    parser.add_argument(
        "--aggressive-mode", type=int, default=0, help="aggressive mode for split"
    )
    parser.add_argument(
        "--path-field",
        type=int,
        default=2,
        help="the field num of wave path after spliting",
    )
    parser.add_argument("--verbose", default=0, type=int, help="Verbose option")
    return parser


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    time_stamps = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        cur_status = "1" if is_speech else "0"
        logging.debug("current frame is %s" % cur_status)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                logging.debug("triggered ait (%.2f)" % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                etime = frame.timestamp + frame.duration
                logging.debug("detriggered at (%.2f)" % etime)
                triggered = False
                logging.info(
                    "find segment from (%.2f - %.2f), length %.2f"
                    % (
                        voiced_frames[0].timestamp,
                        etime,
                        etime - voiced_frames[0].timestamp,
                    )
                )
                time_stamps.append(
                    [
                        voiced_frames[0].timestamp,
                        etime,
                        etime - voiced_frames[0].timestamp,
                    ]
                )
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
    #     logging.info("-(%s)" % (frame.timestamp + frame.duration))
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        etime = voiced_frames[-1].timestamp + voiced_frames[-1].duration
        logging.info(
            "find segment from (%.2f - %.2f), length %.2f"
            % (
                voiced_frames[0].timestamp,
                etime,
                etime - voiced_frames[0].timestamp,
            )
        )
        time_stamps.append(
            [voiced_frames[0].timestamp, etime, etime - voiced_frames[0].timestamp]
        )

    return time_stamps


def main():
    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

    wave_lst = codecs.open(
        os.path.join(args.data_dir, "wav.scp"), "r", encoding="utf-8"
    )
    segment_lst = codecs.open(
        os.path.join(args.data_dir, "segments"), "w", encoding="utf-8"
    )
    vad = webrtcvad.Vad(args.aggressive_mode)

    num_seg = 0
    num_len = 0.0
    for w_i in wave_lst:
        wave_name = w_i.strip().split()[0]
        wave_path = w_i.strip().split()[args.path_field]
        wave_data, sample_rate = read_wave(wave_path)

        frames = list(frame_generator(args.frame_duration, wave_data, sample_rate))
        timestamps = vad_collector(
            sample_rate, args.frame_duration, args.padding_duration, vad, frames
        )

        num_seg += len(timestamps)
        num_len += sum([t[2] for t in timestamps])

        for i, stamp in enumerate(timestamps):
            stime, etime = stamp[0], stamp[1]
            out_name = "%s_%07d_%07d" % (wave_name, stime * 100, etime * 100)
            segment_lst.write("%s %s %.2f %.2f\n" % (out_name, wave_name, stime, etime))

    logging.info(
        "num of segments are %02d, avg time is %.2f" % (num_seg, num_len / num_seg)
    )


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
