#!/usr/bin/env python3
"""Prepare ESPnet data directories for FLEURS, CS-FLEURS and CS-YODAS LID.

Inputs:
  --fleurs_tsv_root  train.tsv / dev.tsv / test.tsv from local/create_fleurs_tsv.py
  --cs_fleurs_root   byan/cs-fleurs checkout (xtts/train, read/test, ...)
  --cs_yodas_root    byan/cs-yodas download (metadata/<lang>.jsonl and audio/)

Labels are ISO 639-3-style codes (eng, ara, cmn, ...). Every data directory
contains wav.scp, text (<ara> <eng>), utt2spk/spk2utt (one speaker per
utterance), utt2lang (the first language, or the atomic pair class in
*_pair directories), utt2langs (all languages), lang2utt and utt2dur.

Directories are named <split>_fleurscs_<data>[_pair]; the constant second
field makes lid.sh treat every test set as in-domain. Train/valid keep
1 s <= duration < 30 s by the metadata duration (CS-YODAS: < 70 s); test sets
are not filtered.
  {train,valid,test}_fleurscs_fleurs       FLEURS only
  {train,valid}_fleurscs_cs                FLEURS + CS-FLEURS, language labels
  {train,valid}_fleurscs_cs_pair           same utterances, one class per pair
  {train,valid}_fleurscs_csyodas           + CS-YODAS base-language/English pairs
  {train,valid}_fleurscs_csyodas_pair      same utterances, one class per pair
  test_fleurscs_cs_{read,xtts1,xtts2,mms}  CS-FLEURS test sets
  test_fleurscs_cs_*_pair                  pair-class references
"""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FLEURS_TO_ISO3: Dict[str, str] = {
    "af_za": "afr",
    "am_et": "amh",
    "ar_eg": "ara",
    "as_in": "asm",
    "ast_es": "ast",
    "az_az": "aze",
    "be_by": "bel",
    "bg_bg": "bul",
    "bn_in": "ben",
    "bs_ba": "bos",
    "ca_es": "cat",
    "ceb_ph": "ceb",
    "ckb_iq": "ckb",
    "cmn_hans_cn": "cmn",
    "cs_cz": "ces",
    "cy_gb": "cym",
    "da_dk": "dan",
    "de_de": "deu",
    "el_gr": "ell",
    "en_us": "eng",
    "es_419": "spa",
    "et_ee": "est",
    "fa_ir": "fas",
    "ff_sn": "ful",
    "fi_fi": "fin",
    "fil_ph": "tgl",
    "fr_fr": "fra",
    "ga_ie": "gle",
    "gl_es": "glg",
    "gu_in": "guj",
    "ha_ng": "hau",
    "he_il": "heb",
    "hi_in": "hin",
    "hr_hr": "hrv",
    "hu_hu": "hun",
    "hy_am": "hye",
    "id_id": "ind",
    "ig_ng": "ibo",
    "is_is": "isl",
    "it_it": "ita",
    "ja_jp": "jpn",
    "jv_id": "jav",
    "ka_ge": "kat",
    "kam_ke": "kam",
    "kea_cv": "kea",
    "kk_kz": "kaz",
    "km_kh": "khm",
    "kn_in": "kan",
    "ko_kr": "kor",
    "ky_kg": "kir",
    "lb_lu": "ltz",
    "lg_ug": "lug",
    "ln_cd": "lin",
    "lo_la": "lao",
    "lt_lt": "lit",
    "luo_ke": "luo",
    "lv_lv": "lav",
    "mi_nz": "mri",
    "mk_mk": "mkd",
    "ml_in": "mal",
    "mn_mn": "mon",
    "mr_in": "mar",
    "ms_my": "zlm",
    "mt_mt": "mlt",
    "my_mm": "mya",
    "nb_no": "nob",
    "ne_np": "nep",
    "nl_nl": "nld",
    "nso_za": "nso",
    "ny_mw": "nya",
    "oc_fr": "oci",
    "om_et": "orm",
    "or_in": "ory",
    "pa_in": "pan",
    "pl_pl": "pol",
    "ps_af": "pus",
    "pt_br": "por",
    "ro_ro": "ron",
    "ru_ru": "rus",
    "sd_in": "snd",
    "sk_sk": "slk",
    "sl_si": "slv",
    "sn_zw": "sna",
    "so_so": "som",
    "sr_rs": "srp",
    "sv_se": "swe",
    "sw_ke": "swh",
    "ta_in": "tam",
    "te_in": "tel",
    "tg_tj": "tgk",
    "th_th": "tha",
    "tr_tr": "tur",
    "uk_ua": "ukr",
    "umb_ao": "umb",
    "ur_pk": "urd",
    "uz_uz": "uzb",
    "vi_vn": "vie",
    "wo_sn": "wol",
    "xh_za": "xho",
    "yo_ng": "yor",
    "yue_hant_hk": "yue",
    "zu_za": "zul",
}
LANGUAGES = frozenset(FLEURS_TO_ISO3.values())

CS_FLEURS_TRAIN_SUBSETS = ("xtts/train",)
CS_FLEURS_TEST_SUBSETS = {  # subset directory -> data directory token
    "read/test": "cs_read",
    "xtts/test1": "cs_xtts1",
    "xtts/test2": "cs_xtts2",
    "mms/test": "cs_mms",
}
CS_FLEURS_DEV_RATIO = 0.02

CS_YODAS_LANGUAGES = {
    "ara": "Arabic",
    "cmn": "Chinese",
    "fra": "French",
    "hin": "Hindi",
    "jpn": "Japanese",
    "rus": "Russian",
}
CS_YODAS_METADATA_SHA256 = {
    "ara": "19254013052100a5c780d2342c21185babee874cec6a7cb5f14c67bae1435abb",
    "cmn": "6cdad8b713ea41a598077dd0fedae14824eb2d6998970132d5b2c885201c7697",
    "fra": "76f415663519b4898ddd0540bbf7b5149aae25f181e9262a39e89aae843fbfda",
    "hin": "2533311b72283f436ec9a0aca687b8f66983bd2c1fdd2574704a3091cba3e526",
    "jpn": "c5fca5676a09f07391aee943996ce6619529bea273b17237f756f1e66c7a5844",
    "rus": "f29381d741c611e54e5ad06d7645f4a9fa837508ea8784006d4f600d70be006a",
}
CS_YODAS_SPLIT_SEED = "cs-yodas-v1"
CS_YODAS_UTTID_RE = re.compile(
    r"(?P<video>.+)_[a-z]{3}_(?P<start>\d{9})_(?P<end>\d{9})$"
)


@dataclass(frozen=True)
class Example:
    """One utterance: id, audio path, language labels and duration (s)."""

    uttid: str
    wav: str
    labels: Tuple[str, ...]
    duration: float


def sanitize_id(text: str) -> str:
    """Replace characters that are not safe in Kaldi-style utterance ids."""
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "utt"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    """Yield the JSON objects of the non-empty lines of a JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fleurs_uttid(lang: str, split: str, row: Dict[str, str]) -> str:
    """Build fleurs_<config>_<split>_<id>_<stem> from stable FLEURS fields.

    <id> is the FLEURS sentence id (reused across languages and recordings)
    and <stem> is the original audio file name without extension.
    """
    stem = Path(row["source_file"]).stem
    return sanitize_id(f"fleurs_{lang}_{split}_{row['id']}_{stem}")


def load_fleurs(tsv_root: Path, split: str) -> List[Example]:
    """Read the FLEURS TSV manifest of one split."""
    examples = []
    with (tsv_root / f"{split}.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            lang = row["lang_id_name"]
            examples.append(
                Example(
                    fleurs_uttid(lang, split, row),
                    row["path"],
                    (FLEURS_TO_ISO3[lang],),
                    float(row["duration"]),
                )
            )
    return examples


def pair_labels(value: str) -> Tuple[str, ...]:
    """Parse a CS-FLEURS language field such as ``ara-eng``."""
    labels = tuple(value.strip().lower().split("-"))
    if len(labels) != 2 or len(set(labels)) != 2 or not set(labels) <= LANGUAGES:
        raise ValueError(f"unexpected CS-FLEURS language pair: {value!r}")
    return labels


def load_cs_fleurs(cs_root: Path, subset: str) -> List[Example]:
    """Read the metadata.jsonl of one CS-FLEURS subset."""
    subset_dir = cs_root / subset
    examples = []
    for row in iter_jsonl(subset_dir / "metadata.jsonl"):
        examples.append(
            Example(
                sanitize_id(f"cs_{subset.replace('/', '_')}_{row['id']}"),
                str((subset_dir / row["file_name"]).resolve()),
                pair_labels(row["language"]),
                float(row["duration"]),
            )
        )
    return examples


def stable_fraction(key: str) -> float:
    """Map a string to a deterministic number in [0, 1)."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16) / 16**12


def split_cs_fleurs(
    examples: Sequence[Example], dev_ratio: float
) -> Tuple[List[Example], List[Example]]:
    """Deterministic utterance-level split by a hash of the utterance id."""
    valid = [ex for ex in examples if stable_fraction(ex.uttid) < dev_ratio]
    valid_ids = {ex.uttid for ex in valid}
    train = [ex for ex in examples if ex.uttid not in valid_ids]
    return train, valid


def load_cs_yodas(yodas_root: Path) -> List[Tuple[Example, str]]:
    """Return (example, video_id) for every exact base-language + English row."""
    records = []
    for lang, name in CS_YODAS_LANGUAGES.items():
        path = yodas_root / "metadata" / f"{lang}.jsonl"
        if sha256_file(path) != CS_YODAS_METADATA_SHA256[lang]:
            raise ValueError(f"CS-YODAS metadata checksum mismatch: {path}")
        for row in iter_jsonl(path):
            languages = row.get("languages")
            if not isinstance(languages, list) or set(languages) != {name, "English"}:
                continue
            if len(languages) != 2:
                continue
            match = CS_YODAS_UTTID_RE.match(row["kaldi_uttid"])
            if match is None:
                raise ValueError(f"cannot parse kaldi_uttid: {row['kaldi_uttid']}")
            duration = (int(match.group("end")) - int(match.group("start"))) / 1000.0
            uttid = re.sub(r"[^A-Za-z0-9_.=-]+", "_", row["id"]).strip("._-")
            records.append(
                (
                    Example(
                        f"csyodas_{lang}_{uttid}",
                        str((yodas_root / "audio" / row["wav_path"]).resolve()),
                        (lang, "eng"),
                        duration,
                    ),
                    match.group("video"),
                )
            )
    return records


def split_cs_yodas(
    records: Sequence[Tuple[Example, str]], train_ratio: float, valid_ratio: float
) -> Dict[str, List[Example]]:
    """Split per base language by video id (80/10/10); a video never crosses."""
    by_lang: Dict[str, Dict[str, List[Example]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ex, video in records:
        by_lang[ex.labels[0]][video].append(ex)
    split: Dict[str, List[Example]] = {"train": [], "valid": [], "test": []}
    for lang, groups in sorted(by_lang.items()):
        items = sorted(
            groups.items(),
            key=lambda kv: hashlib.sha1(
                f"{CS_YODAS_SPLIT_SEED}\t{lang}\t{kv[0]}".encode()
            ).hexdigest(),
        )
        total = sum(len(v) for _, v in items)
        n_train = n_valid = 0
        for _, group in items:
            if n_train < total * train_ratio:
                dst = "train"
                n_train += len(group)
            elif n_valid < total * valid_ratio:
                dst = "valid"
                n_valid += len(group)
            else:
                dst = "test"
            split[dst].extend(group)
    return split


def filter_duration(
    examples: Sequence[Example], min_sec: Optional[float], max_sec: float
) -> List[Example]:
    """Keep min_sec <= duration < max_sec (no lower bound if min_sec is None)."""
    return [
        ex
        for ex in examples
        if (min_sec is None or ex.duration >= min_sec) and ex.duration < max_sec
    ]


def pair_class(labels: Sequence[str]) -> str:
    """One atomic class per language set: xxx-eng for English pairs."""
    if len(labels) == 1:
        return labels[0]
    if "eng" in labels:
        return f"{labels[0] if labels[1] == 'eng' else labels[1]}-eng"
    return "-".join(sorted(labels))


def to_pair_view(examples: Sequence[Example]) -> List[Example]:
    """Replace the labels of every example with its atomic pair class."""
    return [
        Example(ex.uttid, ex.wav, (pair_class(ex.labels),), ex.duration)
        for ex in examples
    ]


def write_data_dir(outdir: Path, examples: Sequence[Example]) -> None:
    """Write one Kaldi-style data directory (see the module docstring)."""
    outdir.mkdir(parents=True, exist_ok=True)
    examples = sorted(examples, key=lambda ex: ex.uttid)
    if len({ex.uttid for ex in examples}) != len(examples):
        raise ValueError(f"duplicate utterance ids in {outdir}")
    lang2utt: Dict[str, List[str]] = defaultdict(list)
    files = {
        name: (outdir / name).open("w", encoding="utf-8")
        for name in ("wav.scp", "text", "utt2spk", "spk2utt", "utt2lang", "utt2langs")
    }
    with (outdir / "utt2dur").open("w", encoding="utf-8") as utt2dur:
        for ex in examples:
            files["wav.scp"].write(f"{ex.uttid} {ex.wav}\n")
            files["text"].write(f"{ex.uttid} {' '.join(f'<{x}>' for x in ex.labels)}\n")
            files["utt2spk"].write(f"{ex.uttid} {ex.uttid}\n")
            files["spk2utt"].write(f"{ex.uttid} {ex.uttid}\n")
            files["utt2lang"].write(f"{ex.uttid} {ex.labels[0]}\n")
            files["utt2langs"].write(f"{ex.uttid} {' '.join(ex.labels)}\n")
            utt2dur.write(f"{ex.uttid} {ex.duration:.6f}\n")
            lang2utt[ex.labels[0]].append(ex.uttid)
    for f in files.values():
        f.close()
    with (outdir / "lang2utt").open("w", encoding="utf-8") as f:
        for lang, utts in sorted(lang2utt.items()):
            f.write(f"{lang} {' '.join(utts)}\n")
    print(f"{outdir}: {len(examples)} utterances, {len(lang2utt)} classes")


def build_views(args: argparse.Namespace) -> Dict[str, List[Example]]:
    """Build all data directories as {name: examples}."""
    min_sec, max_sec = args.min_duration, args.max_duration
    fleurs = {s: load_fleurs(args.fleurs_tsv_root, s) for s in ("train", "dev", "test")}
    views = {
        "train_fleurscs_fleurs": filter_duration(fleurs["train"], min_sec, max_sec),
        "valid_fleurscs_fleurs": filter_duration(fleurs["dev"], min_sec, max_sec),
        "test_fleurscs_fleurs": fleurs["test"],
    }
    if args.cs_fleurs_root is not None:
        cs = [
            ex
            for subset in CS_FLEURS_TRAIN_SUBSETS
            for ex in load_cs_fleurs(args.cs_fleurs_root, subset)
        ]
        cs_train, cs_valid = split_cs_fleurs(cs, CS_FLEURS_DEV_RATIO)
        views["train_fleurscs_cs"] = views["train_fleurscs_fleurs"] + filter_duration(
            cs_train, min_sec, max_sec
        )
        views["valid_fleurscs_cs"] = views["valid_fleurscs_fleurs"] + filter_duration(
            cs_valid, min_sec, max_sec
        )
        for subset, token in CS_FLEURS_TEST_SUBSETS.items():
            views[f"test_fleurscs_{token}"] = load_cs_fleurs(
                args.cs_fleurs_root, subset
            )
    if args.cs_yodas_root is not None:
        yodas = split_cs_yodas(load_cs_yodas(args.cs_yodas_root), 0.8, 0.1)
        for split in ("train", "valid"):
            kept = filter_duration(yodas[split], None, args.max_yodas_duration)
            views[f"{split}_fleurscs_csyodas"] = views[f"{split}_fleurscs_cs"] + kept
    # One atomic class per language set (FLEURS-only sets are single-language)
    for name in [n for n in views if not n.endswith("_fleurs")]:
        views[f"{name}_pair"] = to_pair_view(views[name])
    return views


def get_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("data"))
    parser.add_argument("--fleurs_tsv_root", type=Path, required=True)
    parser.add_argument("--cs_fleurs_root", type=Path, default=None)
    parser.add_argument("--cs_yodas_root", type=Path, default=None)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--max_yodas_duration", type=float, default=70.0)
    return parser


def main() -> None:
    """Write the data directories."""
    args = get_parser().parse_args()
    if args.cs_yodas_root is not None and args.cs_fleurs_root is None:
        raise ValueError("--cs_yodas_root requires --cs_fleurs_root")
    for name, examples in build_views(args).items():
        write_data_dir(args.outdir / name, examples)


if __name__ == "__main__":
    main()
