import argparse
import os
import re

configs = [
    "eng1",
    "eng2",
    "eng3",
    "fra1",
    "fra2",
    "deu1",
    "deu2",
    "rus",
    "swa",
    "swe",
    "jpn",
    "cmn",
    "xty",
]
langs = [
    "eng",
    "eng",
    "eng",
    "fra",
    "fra",
    "deu",
    "deu",
    "rus",
    "swa",
    "swe",
    "jpn",
    "cmn",
    "xty",
]
durations = ["10min", "1h"]


def mean(iter_feat):
    sum = 0
    for v in iter_feat:
        sum += v
    return sum / (len(iter_feat))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expdir", type=str, default="exp")
    parser.add_argument("--asr_tag_prefix", type=str, default="train_asr_fbank")
    parser.add_argument("--inference_model", type=str, default="valid.loss.ave.pth")
    parser.add_argument("--log", type=str, default="mono_train_asr_fbank.log")

    args = parser.parse_args()

    log_file = open("{}/{}".format(args.expdir, args.log), "w", encoding="utf-8")

    if args.inference_model.endswith(".pth"):
        inference_tag = args.inference_model[:-4]
    else:
        inference_tag = args.inference_model

    for dur in durations:
        result_dict = {}
        log_record = "{},".format(dur)
        for i, config in enumerate(configs):
            # Set related folders
            asr_exp = "asr_{}_{}_{}".format(args.asr_tag_prefix, config, dur)
            decoder_folder = "decode_asr_asr_model_{}".format(inference_tag)
            eval_folder = "test_10min_{}".format(config)
            if config in ["cmn", "jpn"]:
                score_folder = "score_wer"
            else:
                score_folder = "score_cer"

            result = os.path.join(
                args.expdir,
                asr_exp,
                decoder_folder,
                eval_folder,
                score_folder,
                "result.txt",
            )

            if not os.path.exists(result):
                raise RuntimeError(
                    "Cannot find result file at {}, might be "
                    "due to unsuccessfully experiments for {}, {}".format(
                        result, config, dur
                    )
                )

            info = open(result, "r", encoding="utf-8")
            result_info = info.read()

            error_rate = re.search("Sum/Avg(.*\n)", result_info)[0].split()[9]
            result_dict[langs[i]] = result_dict.get(langs[i], []) + [float(error_rate)]
            log_record += "{},".format(error_rate)

        avg_error_rate = mean([mean(result_dict[v]) for v in result_dict.keys()])
        log_file.write("Average Error Rate ({}):{}\n".format(dur, avg_error_rate))
        print("Average Error Rate ({}):{}\n".format(dur, avg_error_rate))
        log_file.write(
            "Details:\n{},{}\n{}\n".format("Dur", ",".join(configs), log_record)
        )
        print("Details are saved in {}/{}".format(args.expdir, args.log))
