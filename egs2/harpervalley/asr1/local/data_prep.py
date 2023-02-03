import argparse
import json
import os
import sys
import wave


def load_json(f_path):
    with open(f_path, "r") as f:
        return json.load(f)


def process_data(target_dir, source_dir, audio_dir, filename, min_length):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    agent_wav_path = os.path.join(audio_dir, "agent", filename + ".wav")
    caller_wav_path = os.path.join(audio_dir, "caller", filename + ".wav")
    # exit if the wav files do not exist.
    if not os.path.isfile(agent_wav_path) or not os.path.isfile(agent_wav_path):
        sys.exit()

    with wave.open(agent_wav_path, "rb") as wa, wave.open(caller_wav_path, "rb") as wc:
        wa_length = wa.getnframes() / wa.getframerate()
        wc_length = wc.getnframes() / wc.getframerate()

    with open(
        os.path.join(target_dir, "wav.scp"), "a", encoding="utf-8"
    ) as wavscp, open(
        os.path.join(target_dir, "utt2spk"), "a", encoding="utf-8"
    ) as utt2spk, open(
        os.path.join(target_dir, "segments"), "a", encoding="utf-8"
    ) as segments, open(
        os.path.join(target_dir, "text"), "a", encoding="utf-8"
    ) as text:
        metadata_f = load_json(os.path.join(source_dir, "metadata", filename + ".json"))
        transcript_f = load_json(
            os.path.join(source_dir, "transcript", filename + ".json")
        )

        agent_spk_id = metadata_f["agent"]["speaker_id"]
        caller_spk_id = metadata_f["caller"]["speaker_id"]
        task_type = metadata_f["tasks"][0]["task_type"].replace(" ", "_")
        agent_rec_id = "{}-{}".format(agent_spk_id, filename)
        caller_rec_id = "{}-{}".format(caller_spk_id, filename)

        agent_utt_num = 0
        caller_utt_num = 0
        for v in transcript_f:
            transcript = v["human_transcript"]
            # Throw away utterances with < min_length words or 100 ms
            if len(transcript.split()) < min_length or int(v["duration_ms"]) < 100:
                continue
            begin_ms = int(v["offset_ms"])
            end_ms = begin_ms + int(v["duration_ms"])
            begin_sec = begin_ms / 1000
            end_sec = end_ms / 1000
            if v["speaker_role"] == "agent":
                if end_sec > wa_length:
                    continue
                utt_id = "{}_{}_{}".format(agent_rec_id, begin_ms, end_ms)
                utt2spk.write("{} {}\n".format(utt_id, agent_spk_id))
                segments.write(
                    "{} {} {} {}\n".format(utt_id, agent_rec_id, begin_sec, end_sec)
                )
                agent_utt_num += 1
            else:
                if end_sec > wc_length:
                    continue
                utt_id = "{}_{}_{}".format(caller_rec_id, begin_ms, end_ms)
                utt2spk.write("{} {}\n".format(utt_id, caller_spk_id))
                segments.write(
                    "{} {} {} {}\n".format(utt_id, caller_rec_id, begin_sec, end_sec)
                )
                caller_utt_num += 1
            text.write("{} {} {}\n".format(utt_id, task_type, transcript))

        # write wav.scp only if utterances exist
        if agent_utt_num > 0:
            wavscp.write("{} {}\n".format(agent_rec_id, agent_wav_path))
        if caller_utt_num > 0:
            wavscp.write("{} {}\n".format(caller_rec_id, caller_wav_path))


parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", type=str, default="data/tmp")
parser.add_argument("--source_dir", type=str, required=True, help="Path to source data")
parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio data")
parser.add_argument("--filename", type=str, required=True, help="filename")
parser.add_argument("--min_length", type=int, default=4)

args = parser.parse_args()

process_data(
    args.target_dir, args.source_dir, args.audio_dir, args.filename, args.min_length
)
