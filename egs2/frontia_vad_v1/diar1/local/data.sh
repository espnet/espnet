#!/usr/bin/env bash
# Prepare the VAD data directories of this recipe.
#
# Every corpus gives one directory per split, data/<corpus>_<split>; the
# training and validation sets are the concatenation over --corpora:
#   data/train  = <corpus>_train for every corpus
#   data/dev    = <corpus>_dev   for every corpus
#   data/<corpus>_eval are kept separate as test sets.
# Evaluation-only corpora have no train split and stay out of data/train and
# data/dev: their dev and eval splits are test sets. They are DiPCo, and the
# corpora with published VAD results (local/prep_eval_corpora.py): ami_mh (AMI
# Mix-Headset, BUT setup), alimeeting, aishell4, voxconverse, earnings21,
# ava_speech and ten_vad; those with a single test split only have <corpus>_eval.
# Labels are utterance level: the union of all speakers' segments, written as
# a single pseudo speaker per recording (diar.sh --num_spk 1).

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

corpora="ami,chime6,notsofar,icsi,dipco"  # comma separated: survives diar.sh word splitting
splits="train,dev,eval"

log "$0 $*"
. utils/parse_options.sh
. ./path.sh
. ./db.sh

ami_lists=../../ami/asr1/local   # full-corpus-ASR partition of the AMI recipe
eval_only="dipco ami_mh alimeeting aishell4 voxconverse earnings21 ava_speech ten_vad"
eval_split_only="aishell4 earnings21 ava_speech ten_vad"  # no dev split

# The first of the given paths that exists, or nothing.
first_existing() {
    local p
    for p in "$@"; do
        if [ -e "${p}" ]; then
            echo "${p}"
            return
        fi
    done
}

# A directory holding <name>, or <name> unpacked into downloads/ from <name>.tar.gz
# found in the directory or its ARCHIVES/ (only the members matching <members>).
unpacked() {
    local root=$1 name=$2 members=$3 archive
    if [ -d "${root}/${name}" ]; then
        echo "${root}/${name}"
        return
    fi
    if [ ! -d "downloads/${name}" ]; then
        archive=$(first_existing "${root}/${name}.tar.gz" "${root}/ARCHIVES/${name}.tar.gz")
        [ -n "${archive}" ] || { log "No ${name} or ${name}.tar.gz in ${root}" >&2; exit 1; }
        mkdir -p downloads
        log "Unpacking ${members} from ${archive} into downloads/" >&2
        tar xzf "${archive}" -C downloads --wildcards "${members}"
    fi
    echo "downloads/${name}"
}

for corpus in ${corpora//,/ }; do
    for split in ${splits//,/ }; do
        if [[ " ${eval_only} " == *" ${corpus} "* ]] && [ "${split}" = train ]; then
            continue
        fi
        if [[ " ${eval_split_only} " == *" ${corpus} "* ]] && [ "${split}" != eval ]; then
            continue
        fi
        out=data/${corpus}_${split}
        case ${corpus} in
        ami)
            [ -n "${AMI}" ] || { log "Set AMI in db.sh"; exit 1; }
            # <meeting>/audio/*.wav sits in ${AMI} or ${AMI}/wav_db (lhotse layout);
            # the manual annotations may still be the release zip
            ami_wavs=${AMI}; [ -d "${AMI}/wav_db" ] && ami_wavs=${AMI}/wav_db
            ami_annot=${AMI}/annotations
            if [ ! -d "${ami_annot}" ]; then
                ami_zips=("${AMI}"/ami_public_manual_*.zip)  # sorted: the last is the newest
                ami_annot=${ami_zips[${#ami_zips[@]}-1]}
            fi
            python3 local/prep_ami.py "${ami_wavs}" "${ami_annot}" \
                "${ami_lists}/split_${split}.orig" "${out}"
            ;;
        chime6)
            [ -n "${CHIME6}" ] || { log "Set CHIME6 in db.sh"; exit 1; }
            python3 local/prep_chime6.py "${CHIME6}" "${split}" "${out}"
            ;;
        notsofar)
            [ -n "${NOTSOFAR1}" ] || { log "Set NOTSOFAR1 in db.sh"; exit 1; }
            python3 local/prep_notsofar.py "${NOTSOFAR1}" "${split}" "${out}"
            ;;
        icsi)
            [ -n "${ICSI}" ] || { log "Set ICSI in db.sh"; exit 1; }
            # <meeting>/*.sph sits in ${ICSI}/speech or ${ICSI}/ICSI/speech
            icsi_speech=${ICSI}/speech; [ -d "${icsi_speech}" ] || icsi_speech=${ICSI}/ICSI/speech
            # ICSI audio is SPHERE; write the sph2pipe that exists, even off PATH
            sph2pipe=$(command -v sph2pipe || echo "${MAIN_ROOT}/tools/sph2pipe/sph2pipe")
            python3 local/prep_icsi.py --sph2pipe "${sph2pipe}" "${icsi_speech}" \
                "${ICSI}/ICSI_core_NXT.zip" "local/icsi_split_${split}.txt" "${out}"
            ;;
        dipco)
            [ -n "${DIPCO}" ] || { log "Set DIPCO in db.sh"; exit 1; }
            python3 local/prep_dipco.py "${DIPCO}" "${split}" "${out}"
            ;;
        ami_mh)
            [ -n "${AMI}" ] || { log "Set AMI in db.sh"; exit 1; }
            setup=${AMI_DIAR_SETUP:-downloads/AMI-diarization-setup}
            if [ ! -d "${setup}" ]; then
                git clone https://github.com/BUTSpeechFIT/AMI-diarization-setup "${setup}"
                git -C "${setup}" checkout 2509d8933721023fab4def2618aabd5c28eb82e9
            fi
            python3 local/prep_eval_corpora.py ami_mh "${split}" "${out}" "${AMI}" "${setup}"
            ;;
        alimeeting)
            [ -n "${ALIMEETING}" ] || { log "Set ALIMEETING in db.sh"; exit 1; }
            name=$([ "${split}" = dev ] && echo Eval_Ali || echo Test_Ali)
            dir=$(unpacked "${ALIMEETING}" "${name}" "${name}/${name}_far/*")
            python3 local/prep_eval_corpora.py alimeeting "${split}" "${out}" "$(dirname "${dir}")"
            ;;
        aishell4)
            [ -n "${AISHELL4}" ] || { log "Set AISHELL4 in db.sh"; exit 1; }
            dir=$(unpacked "${AISHELL4}" test "test/*")
            python3 local/prep_eval_corpora.py aishell4 eval "${out}" "${dir}"
            ;;
        voxconverse)
            [ -n "${VOXCONVERSE}" ] || { log "Set VOXCONVERSE in db.sh"; exit 1; }
            part=$([ "${split}" = dev ] && echo dev || echo test)
            audio=$(first_existing "${VOXCONVERSE}/voxconverse_${part}_wav" \
                "${VOXCONVERSE}/voxconverse_${part}_wav.zip" \
                "${VOXCONVERSE}/ARCHIVES/voxconverse_${part}_wav.zip")
            [ -n "${audio}" ] || { log "No voxconverse_${part}_wav(.zip) in ${VOXCONVERSE}"; exit 1; }
            repo=${VOXCONVERSE}/voxconverse
            if [ ! -d "${repo}/${part}" ]; then
                repo=downloads/voxconverse
                [ -d "${repo}" ] || git clone https://github.com/joonson/voxconverse "${repo}"
            fi
            python3 local/prep_eval_corpora.py voxconverse "${split}" "${out}" "${audio}" "${repo}/${part}"
            ;;
        earnings21)
            [ -n "${EARNINGS21}" ] || { log "Set EARNINGS21 in db.sh"; exit 1; }
            python3 local/prep_eval_corpora.py earnings21 eval "${out}" "${EARNINGS21}"
            ;;
        ava_speech)
            [ -n "${AVA_SPEECH}" ] || { log "Set AVA_SPEECH in db.sh"; exit 1; }
            python3 local/prep_eval_corpora.py ava_speech eval "${out}" "${AVA_SPEECH}"
            ;;
        ten_vad)
            [ -n "${TEN_VAD}" ] || { log "Set TEN_VAD in db.sh"; exit 1; }
            python3 local/prep_eval_corpora.py ten_vad eval "${out}" "${TEN_VAD}"
            ;;
        *)
            log "Unknown corpus: ${corpus}"; exit 1 ;;
        esac
        utils/utt2spk_to_spk2utt.pl "${out}/utt2spk" > "${out}/spk2utt"
        utils/fix_data_dir.sh "${out}"
    done
done

for split in train dev; do
    dirs=""
    for corpus in ${corpora//,/ }; do
        [[ " ${eval_only} " == *" ${corpus} "* ]] && continue
        if [ -s "data/${corpus}_${split}/wav.scp" ]; then
            dirs+="data/${corpus}_${split} "
        fi
    done
    if [ -n "${dirs}" ]; then
        # shellcheck disable=SC2086
        utils/combine_data.sh --skip_fix true "data/${split}" ${dirs}
        utils/fix_data_dir.sh "data/${split}"
    fi
done

# Noise for on-the-fly augmentation (--noise_scp). MUSAN speech/ is left out:
# babble added to a non-speech region would be labelled non-speech.
if [ -n "${MUSAN}" ] || [ -n "${CHIME3_BACKGROUND}" ]; then
    {
        if [ -n "${MUSAN}" ]; then
            find "${MUSAN}/noise" "${MUSAN}/music" -name '*.wav'
        fi
        if [ -n "${CHIME3_BACKGROUND}" ]; then
            find "${CHIME3_BACKGROUND}" -name '*.wav'
        fi
    } | sort | awk '{n=$0; sub(".*/", "", n); sub("\\.wav$", "", n); print n, $0}' \
        > data/noise.scp
    log "data/noise.scp: $(wc -l < data/noise.scp) noise files"
else
    log "MUSAN and CHIME3_BACKGROUND unset in db.sh: no data/noise.scp"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
