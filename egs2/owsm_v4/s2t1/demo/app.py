import os

import gradio as gr
import librosa
import torch

try:  # ZeroGPU's decorator, which exists only on Hugging Face's runners
    import spaces
except ImportError:  # running locally: the decorator does nothing

    class spaces:  # noqa: N801 - mirrors the module it stands in for
        @staticmethod
        def GPU(func):
            return func


from espnet2.bin.s2t_inference_language import Speech2Language
from espnet2.bin.s2t_inference import Speech2Text as ARSpeech2Text
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch as CTCSpeech2Text


TITLE="Open Whisper-style Speech Model V4 from CMU WAVLab"

DESCRIPTION='''
OWSM (pronounced as "awesome") is a series of Open Whisper-style Speech Models from [CMU WAVLab](https://www.wavlab.org/).
We reproduce Whisper-style training using publicly available data and an open-source toolkit [ESPnet](https://github.com/espnet/espnet).
For more details, please check our [website](https://www.wavlab.org/activities/2024/owsm/).
'''

ARTICLE = '''
The latest demo uses OWSM v4 based on [E-Branchformer](https://arxiv.org/abs/2210.00077).
OWSM v4 medium model has 1.02B parameters and is trained on 320k hours of labelled data (290k for ASR, 30k for ST).
OWSM-V4 CTC model has 1.01B parameters and is trained on the same dataset as the medium model.
They supports various speech-to-text tasks:
- Speech recognition in 151 languages
- Any-to-any language speech translation
- Utterance-level timestamp prediction
- Long-form transcription
- Language identification

Additionally, OWSM v4 applies 8 times subsampling (instead of 4 times in OWSM v3.1) to the log Mel features, leading to a final resolution of 80 ms in the encoder. When running inference, we recommend setting maxlenratio=1.0 (default) instead of smaller values.

As a demo, the input speech should not exceed 2 minutes. We also limit the maximum number of tokens to be generated.
Please try our [Colab demo](https://colab.research.google.com/drive/1zKI3ZY_OtZd6YmVeED6Cxy1QwT1mqv9O?usp=sharing) if you want to explore more features.

**Disclaimer:** OWSM has not been thoroughly evaluated in all tasks. Due to limited training data, it may not perform well for certain languages.

Please consider citing the following papers if you find our work helpful.

```
@inproceedings{owsm-v4,
  title={{OWSM} v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning},
  author={Yifan Peng and Shakeel Muhammad and Yui Sudo and William Chen and Jinchuan Tian and Chyi-Jiunn Lin and Shinji Watanabe},
  booktitle={Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2025},
}
@inproceedings{peng2024owsm31,
  title={OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer},
  author={Yifan Peng and Jinchuan Tian and William Chen and Siddhant Arora and Brian Yan and Yui Sudo and Muhammad Shakeel and Kwanghee Choi and Jiatong Shi and Xuankai Chang and Jee-weon Jung and Shinji Watanabe},
  booktitle={Proc. INTERSPEECH},
  year={2024}
}
@inproceedings{peng2023owsm,
  title={Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data},
  author={Yifan Peng and Jinchuan Tian and Brian Yan and Dan Berrebbi and Xuankai Chang and Xinjian Li and Jiatong Shi and Siddhant Arora and William Chen and Roshan Sharma and Wangyou Zhang and Yui Sudo and Muhammad Shakeel and Jee-weon Jung and Soumi Maiti and Shinji Watanabe},
  booktitle={Proc. ASRU},
  year={2023}
}
@inproceedings{owsm-ctc,
    title = "{OWSM}-{CTC}: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification",
    author = "Peng, Yifan  and
      Sudo, Yui  and
      Shakeel, Muhammad  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)",
    year = "2024",
    month= {8},
    url = "https://aclanthology.org/2024.acl-long.549",
}

```
'''

# CUDA on the Space, whatever is available elsewhere; DEVICE overrides both.
device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

s2l = Speech2Language.from_pretrained(
    model_tag=f"espnet/owsm_v4_medium_1B",
    device=device,
    nbest=1,
)

s2t_ar = ARSpeech2Text.from_pretrained(
    model_tag=f"espnet/owsm_v4_medium_1B",
    device=device,
    beam_size=5,
    ctc_weight=0.0,
    maxlenratio=0.0,
    # below are default values which can be overwritten in __call__
    lang_sym="<eng>",
    task_sym="<asr>",
    predict_time=False,
)

# CTC looks okay.
s2t_ctc = CTCSpeech2Text.from_pretrained(
    model_tag=f"espnet/owsm_ctc_v4_1B",
    device=device,
    lang_sym="<eng>",
    task_sym="<asr>",
    predict_time=False,
)


iso_codes = ['abk', 'afr', 'amh', 'ara', 'asm', 'ast', 'aze', 'bak', 'bas', 'bel', 'ben', 'bos', 'bre', 'bul', 'cat', 'ceb', 'ces', 'chv', 'ckb', 'cmn', 'cnh', 'cym', 'dan', 'deu', 'dgd', 'div', 'ell', 'eng', 'epo', 'est', 'eus', 'fas', 'fil', 'fin', 'fra', 'frr', 'ful', 'gle', 'glg', 'grn', 'guj', 'hat', 'hau', 'heb', 'hin', 'hrv', 'hsb', 'hun', 'hye', 'ibo', 'ina', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kab', 'kam', 'kan', 'kat', 'kaz', 'kea', 'khm', 'kin', 'kir', 'kmr', 'kor', 'lao', 'lav', 'lga', 'lin', 'lit', 'ltz', 'lug', 'luo', 'mal', 'mar', 'mas', 'mdf', 'mhr', 'mkd', 'mlt', 'mon', 'mri', 'mrj', 'mya', 'myv', 'nan', 'nep', 'nld', 'nno', 'nob', 'npi', 'nso', 'nya', 'oci', 'ori', 'orm', 'ory', 'pan', 'pol', 'por', 'pus', 'quy', 'roh', 'ron', 'rus', 'sah', 'sat', 'sin', 'skr', 'slk', 'slv', 'sna', 'snd', 'som', 'sot', 'spa', 'srd', 'srp', 'sun', 'swa', 'swe', 'swh', 'tam', 'tat', 'tel', 'tgk', 'tgl', 'tha', 'tig', 'tir', 'tok', 'tpi', 'tsn', 'tuk', 'tur', 'twi', 'uig', 'ukr', 'umb', 'urd', 'uzb', 'vie', 'vot', 'wol', 'xho', 'yor', 'yue', 'zho', 'zul']
lang_names = ['Abkhazian', 'Afrikaans', 'Amharic', 'Arabic', 'Assamese', 'Asturian', 'Azerbaijani', 'Bashkir', 'Basa (Cameroon)', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Catalan', 'Cebuano', 'Czech', 'Chuvash', 'Central Kurdish', 'Mandarin Chinese', 'Hakha Chin', 'Welsh', 'Danish', 'German', 'Dagaari Dioula', 'Dhivehi', 'Modern Greek (1453-)', 'English', 'Esperanto', 'Estonian', 'Basque', 'Persian', 'Filipino', 'Finnish', 'French', 'Northern Frisian', 'Fulah', 'Irish', 'Galician', 'Guarani', 'Gujarati', 'Haitian', 'Hausa', 'Hebrew', 'Hindi', 'Croatian', 'Upper Sorbian', 'Hungarian', 'Armenian', 'Igbo', 'Interlingua (International Auxiliary Language Association)', 'Indonesian', 'Icelandic', 'Italian', 'Javanese', 'Japanese', 'Kabyle', 'Kamba (Kenya)', 'Kannada', 'Georgian', 'Kazakh', 'Kabuverdianu', 'Khmer', 'Kinyarwanda', 'Kirghiz', 'Northern Kurdish', 'Korean', 'Lao', 'Latvian', 'Lungga', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Ganda', 'Luo (Kenya and Tanzania)', 'Malayalam', 'Marathi', 'Masai', 'Moksha', 'Eastern Mari', 'Macedonian', 'Maltese', 'Mongolian', 'Maori', 'Western Mari', 'Burmese', 'Erzya', 'Min Nan Chinese', 'Nepali (macrolanguage)', 'Dutch', 'Norwegian Nynorsk', 'Norwegian Bokmål', 'Nepali (individual language)', 'Pedi', 'Nyanja', 'Occitan (post 1500)', 'Oriya (macrolanguage)', 'Oromo', 'Odia', 'Panjabi', 'Polish', 'Portuguese', 'Pushto', 'Ayacucho Quechua', 'Romansh', 'Romanian', 'Russian', 'Yakut', 'Santali', 'Sinhala', 'Saraiki', 'Slovak', 'Slovenian', 'Shona', 'Sindhi', 'Somali', 'Southern Sotho', 'Spanish', 'Sardinian', 'Serbian', 'Sundanese', 'Swahili (macrolanguage)', 'Swedish', 'Swahili (individual language)', 'Tamil', 'Tatar', 'Telugu', 'Tajik', 'Tagalog', 'Thai', 'Tigre', 'Tigrinya', 'Toki Pona', 'Tok Pisin', 'Tswana', 'Turkmen', 'Turkish', 'Twi', 'Uighur', 'Ukrainian', 'Umbundu', 'Urdu', 'Uzbek', 'Vietnamese', 'Votic', 'Wolof', 'Xhosa', 'Yoruba', 'Yue Chinese', 'Chinese', 'Zulu']

task_codes = ['asr', 'st_ara', 'st_cat', 'st_ces', 'st_cym', 'st_deu', 'st_eng', 'st_est', 'st_fas', 'st_fra', 'st_ind', 'st_ita', 'st_jpn', 'st_lav', 'st_mon', 'st_nld', 'st_por', 'st_ron', 'st_rus', 'st_slv', 'st_spa', 'st_swe', 'st_tam', 'st_tur', 'st_vie', 'st_zho']
task_names = ['Automatic Speech Recognition', 'Translate to Arabic', 'Translate to Catalan', 'Translate to Czech', 'Translate to Welsh', 'Translate to German', 'Translate to English', 'Translate to Estonian', 'Translate to Persian', 'Translate to French', 'Translate to Indonesian', 'Translate to Italian', 'Translate to Japanese', 'Translate to Latvian', 'Translate to Mongolian', 'Translate to Dutch', 'Translate to Portuguese', 'Translate to Romanian', 'Translate to Russian', 'Translate to Slovenian', 'Translate to Spanish', 'Translate to Swedish', 'Translate to Tamil', 'Translate to Turkish', 'Translate to Vietnamese', 'Translate to Chinese']

model_names = [
    "owsm_ctc_v4_1B",
    "owsm_v4_medium_1B",
]

lang2code = dict(
    [('Unknown', 'none')] + sorted(list(zip(lang_names, iso_codes)), key=lambda x: x[0])
)
task2code = dict(sorted(list(zip(task_names, task_codes)), key=lambda x: x[0]))

code2lang = dict([(v, k) for k, v in lang2code.items()])


# Copied from Whisper utils
def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


@spaces.GPU
def predict(audio_path, src_lang: str, task: str, model_name: str, beam_size, long_form: bool, text_prev: str,):
    task_sym = f'<{task2code[task]}>'
    
    if model_name == "owsm_ctc_v4_1B":
        s2t = s2t_ctc
    elif model_name == "owsm_v4_medium_1B":
        s2t = s2t_ar
    else:
        raise RuntimeError("Not Supported Model.")

    if "ctc" not in model_name:
        s2t.beam_search.beam_size = int(beam_size)
    
    # Our model is trained on 30s and 16kHz
    speech, rate = librosa.load(audio_path, sr=16000) # speech has shape (len,); resample to 16k Hz

    lang_code = lang2code[src_lang]
    if lang_code == 'none':
        # Detect language using the first 30s of speech
        lang_code = s2l(speech)[0][0].strip()[1:-1]
    lang_sym = f'<{lang_code}>'

    # ASR or ST
    if long_form:
        try:
            s2t.maxlenratio = -300
            utts = s2t.decode_long(
                speech,
                condition_on_prev_text=False,
                init_text=text_prev,
                end_time_threshold="<29.00>",
                lang_sym=lang_sym,
                task_sym=task_sym,
            )

            text = []
            for t1, t2, res in utts:
                text.append(f"[{format_timestamp(seconds=t1)} --> {format_timestamp(seconds=t2)}] {res}")
            text = '\n'.join(text)

            return code2lang[lang_code], text
        except:
            print("An exception occurred in long-form decoding. Fall back to standard decoding (only first 30s)")

    #s2t.maxlenratio = -min(300, int((len(speech) / rate) * 10))  # assuming 10 tokens per second
    if len(text_prev) == 0:
        text_prev = "<na>"

    text = s2t(speech, text_prev, lang_sym=lang_sym, task_sym=task_sym)[0][-2]

    return code2lang[lang_code], text


demo = gr.Interface(
    predict,
    inputs=[
        gr.Audio(type="filepath", label="Input Speech (<120s)", max_length=120, sources=["microphone", "upload"], show_download_button=True, show_share_button=True,),
        gr.Dropdown(choices=list(lang2code), value="English", label="Language", info="Language of input speech. Select 'Unknown' (1st option) to detect it automatically."),
        gr.Dropdown(choices=list(task2code), value="Automatic Speech Recognition", label="Task", info="Task to perform on input speech."),
        gr.Dropdown(choices=list(model_names), value="owsm_ctc_v4_1B", label="Model", info="OWSM V4 model to use for recognition."),
        gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Beam Size", info="Beam size used in beam search."),
        gr.Checkbox(label="Long Form (Experimental)", info="Perform long-form decoding. If an exception happens, it will fall back to standard decoding on the initial 30s."),
        gr.Text(label="Text Prompt (Optional)", info="Generation will be conditioned on this prompt if provided"),
    ],
    outputs=[
        gr.Text(label="Predicted Language", info="Language identification is performed if language is unknown."),
        gr.Text(label="Predicted Text", info="Best hypothesis."),
    ],
    title=TITLE,
    description=DESCRIPTION,
    article=ARTICLE,
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch(
        show_api=False,
        share=True,
        ssr_mode=True,
    )
