#!/usr/bin/env python
from pathlib import Path

CONF_DIR = Path("conf/experiments")
BABEL_LANGS_OF_INTEREST = frozenset("101 103 107 203 206 307 402 404".split())
GLOBALPHONE_LANGS_OF_INTEREST = frozenset(
    "Arabic Czech French Korean Mandarin Spanish Thai".split()
)

CONF_TEMPLATE = """
# BABEL TRAIN:
# Amharic - 307
# Bengali - 103
# Cantonese - 101
# Javanese - 402
# Vietnamese - 107
# Zulu - 206
# BABEL TEST:
# Georgian - 404
# Lao - 203
babel_langs="{BABEL_LANGS}"
babel_recog="{BABEL_RECOG_LANGS}"
gp_langs="{GLOBALPHONE_LANGS}"
gp_recog="{GLOBALPHONE_RECOG_LANGS}"
mboshi_train={MBOSHI_TRAIN}
mboshi_recog={MBOSHI_RECOG}
gp_romanized=false
ipa_transcript={USE_IPA}
"""

CONF_DIR.mkdir(parents=True, exist_ok=True)

# Monolingual schemes
for babel_lang in BABEL_LANGS_OF_INTEREST:
    config = CONF_TEMPLATE.format(
        BABEL_LANGS=babel_lang,
        BABEL_RECOG_LANGS=babel_lang,
        GLOBALPHONE_LANGS="",
        GLOBALPHONE_RECOG_LANGS="",
        MBOSHI_TRAIN="false",
        MBOSHI_RECOG="false",
        USE_IPA="true",
    )
    (CONF_DIR / f"monolingual-{babel_lang}-ipa.conf").write_text(config)
for gp_lang in GLOBALPHONE_LANGS_OF_INTEREST:
    config = CONF_TEMPLATE.format(
        BABEL_LANGS="",
        BABEL_RECOG_LANGS="",
        GLOBALPHONE_LANGS=gp_lang,
        GLOBALPHONE_RECOG_LANGS=gp_lang,
        MBOSHI_TRAIN="false",
        MBOSHI_RECOG="false",
        USE_IPA="true",
    )
    (CONF_DIR / f"monolingual-{gp_lang}-ipa.conf").write_text(config)
# MBOSHI
config = CONF_TEMPLATE.format(
    BABEL_LANGS="",
    BABEL_RECOG_LANGS="",
    GLOBALPHONE_LANGS="",
    GLOBALPHONE_RECOG_LANGS="",
    MBOSHI_TRAIN="true",
    MBOSHI_RECOG="true",
    USE_IPA="true",
)
(CONF_DIR / f"monolingual-mboshi-ipa.conf").write_text(config)

# Leave-one-out schemes
for babel_lang in BABEL_LANGS_OF_INTEREST:
    config = CONF_TEMPLATE.format(
        BABEL_LANGS=" ".join(BABEL_LANGS_OF_INTEREST - {babel_lang}),
        BABEL_RECOG_LANGS=babel_lang,
        GLOBALPHONE_LANGS=" ".join(GLOBALPHONE_LANGS_OF_INTEREST),
        GLOBALPHONE_RECOG_LANGS="",
        MBOSHI_TRAIN="true",
        MBOSHI_RECOG="false",
        USE_IPA="true",
    )
    (CONF_DIR / f"oneout-{babel_lang}-ipa.conf").write_text(config)
for gp_lang in GLOBALPHONE_LANGS_OF_INTEREST:
    config = CONF_TEMPLATE.format(
        BABEL_LANGS=" ".join(BABEL_LANGS_OF_INTEREST),
        BABEL_RECOG_LANGS="",
        GLOBALPHONE_LANGS=" ".join(GLOBALPHONE_LANGS_OF_INTEREST - {gp_lang}),
        GLOBALPHONE_RECOG_LANGS=gp_lang,
        MBOSHI_TRAIN="true",
        MBOSHI_RECOG="false",
        USE_IPA="true",
    )
    (CONF_DIR / f"oneout-{gp_lang}-ipa.conf").write_text(config)
# MBOSHI
config = CONF_TEMPLATE.format(
    BABEL_LANGS=" ".join(BABEL_LANGS_OF_INTEREST),
    BABEL_RECOG_LANGS="",
    GLOBALPHONE_LANGS=" ".join(GLOBALPHONE_LANGS_OF_INTEREST),
    GLOBALPHONE_RECOG_LANGS="",
    MBOSHI_TRAIN="false",
    MBOSHI_RECOG="true",
    USE_IPA="true",
)
(CONF_DIR / f"oneout-mboshi-ipa.conf").write_text(config)

# Train-all-test-all scheme
config = CONF_TEMPLATE.format(
    BABEL_LANGS=" ".join(BABEL_LANGS_OF_INTEREST),
    BABEL_RECOG_LANGS=" ".join(BABEL_LANGS_OF_INTEREST),
    GLOBALPHONE_LANGS=" ".join(GLOBALPHONE_LANGS_OF_INTEREST),
    GLOBALPHONE_RECOG_LANGS=" ".join(GLOBALPHONE_LANGS_OF_INTEREST),
    MBOSHI_TRAIN="true",
    MBOSHI_RECOG="true",
    USE_IPA="true",
)
(CONF_DIR / f"all-ipa.conf").write_text(config)
