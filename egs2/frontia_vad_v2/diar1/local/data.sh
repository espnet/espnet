#!/usr/bin/env bash
# Prepare all data of frontia_vad_v2: the training and validation dumps, the
# augmentation lists, the test sets, and a report per corpus.
#
# The training corpora are read straight from their archives under ${VAD_CORPORA}
# (db.sh) and written into 16 kHz Omni-IO dumps, one per corpus and split, with
# labels and domains (local/v2/vad_dump.py). Outputs:
#   dump/raw/train, dump/raw/dev    all training corpora, combined, with
#                                   utt2category (domain) and domains.tsv
#   data/<corpus>_{dev,eval}         the test sets (Kaldi directories; diar.sh
#                                   stage 2 dumps them)
#   data/noise.scp, data/rir.scp     augmentation noise and room impulse responses
#   dump/v2/reports/<corpus>.json    what was read, kept and left out, and why
#
# Stages:
#   0  download the corpora listed in local/v2/downloads/*.tsv (verified by size
#      and checksum; Common Voice, OpenYAP and the v1 corpora: see README.md)
#   1  v1 meeting corpora (AMI, ICSI, CHiME-6, NOTSOFAR-1) and the test sets, as
#      Kaldi directories (local/data_v1.sh); then the training splits of the
#      meeting corpora as v2 dumps (local/v2/from_kaldi.py)
#   2  augmentation lists: noise (local/v2/noise_list.py) and room impulse
#      responses (local/v2/rir_list.py)
#   3  every corpus in --corpora from its archives (local/v2/prep.py), --nshards
#      jobs each
#   4  CTC forced alignment of the transcribed corpora (label source "A") with
#      OWSM-CTC at --align_revision, one GPU job per dump (local/v2/align.py)
#   5  combine into dump/raw/{train,dev} (local/v2/combine.py) and write the
#      corpus reports (local/v2/corpus_report.py)
# Each stage skips the corpora it has finished (.done / .aligned markers), so a
# stage can be rerun, or run for some corpora at a time (--corpora).
#
# Everything that changes the output is an option below or is in local/v2:
# adapters.py (corpora left out, reader options, hour caps), registry.py and
# adapters_*.py (licence filters, splits, labels), prep.py (--pack_seconds).
# --nshards also changes the output: recordings are packed within a shard.

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0
stop_stage=5
corpora=""            # comma separated corpus names; empty: adapters.training_corpora()
download_jobs=8       # parallel downloads in stage 0
v1_train="ami,chime6,notsofar,icsi"
v1_test="dipco,ami_mh,alimeeting,aishell4,voxconverse,earnings21,ava_speech,ten_vad"
v1_nshards=8          # jobs per v1 corpus split in stage 1
nshards=8             # jobs per corpus in stage 3 (changes the packing; keep 8)
pack_seconds=30       # recordings shorter than this are packed (training chunks: 10 s)
align_model=espnet/owsm_ctc_v4_1B
align_revision=6075bf3d42e6037d1ef7abd7c5558ade886e6eaa
align_min_pause=0.2         # pauses between words shorter than this are speech
align_min_word_score=-10.0  # a recording with a word scored lower is dropped
dumpdir=dump/v2

log "$0 $*"
. utils/parse_options.sh
. ./path.sh
. ./db.sh
. ./cmd.sh

[ -n "${VAD_CORPORA:-}" ] || {
    log "Set VAD_CORPORA (the root of the frontia_vad_v2 training corpora) in db.sh"
    exit 1
}
v2=local/v2

if [ -z "${corpora}" ]; then
    corpora=$(python3 -c "import sys; sys.path.insert(0, '${v2}'); \
import adapters; print(','.join(adapters.training_corpora()))")
fi
# the corpus directories the corpora are read from
directories=$(python3 -c "import sys; sys.path.insert(0, '${v2}'); import adapters; \
print(','.join(sorted({adapters.ADAPTERS[c].directory for c in '${corpora}'.split(',') \
if adapters.ADAPTERS[c].directory})))")

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: download into ${VAD_CORPORA}: ${directories}"
    # libriheavy holds the transcripts of librilight
    extra=""
    [[ ",${directories}," == *",librilight,"* ]] && extra=",libriheavy"
    python3 ${v2}/download.py --root "${VAD_CORPORA}" \
        --corpora "${directories}${extra}" --jobs ${download_jobs}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: v1 meeting corpora and the test sets"
    local/data_v1.sh --corpora "${v1_train},${v1_test}" --splits train,dev,eval
    declare -A licence=(
        [ami]="CC BY 4.0" [chime6]="CC BY-SA 4.0 (SA to be removed)"
        [notsofar]="CC BY 4.0" [icsi]="CC BY 4.0"
    )
    for corpus in ${v1_train//,/ }; do
        for split in train dev; do
            out=${dumpdir}/${corpus}/${split}
            [ -f "${out}/.done" ] && continue
            ${train_cmd} JOB=1:${v1_nshards} "${dumpdir}/log/${corpus}_${split}.JOB.log" \
                python3 ${v2}/from_kaldi.py "data/${corpus}_${split}" "${out}" \
                    --corpus "${corpus}" --domain meeting --licence "${licence[${corpus}]}" \
                    --shard "\$((JOB-1))" --nshards ${v1_nshards}
            touch "${out}/.done"
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: augmentation noise and room impulse responses"
    mkdir -p data
    # licence-filtered MUSAN noise and music without vocals, CHiME-3 backgrounds
    # (replaces the list local/data_v1.sh wrote)
    python3 ${v2}/noise_list.py "${VAD_CORPORA}/musan" "${VAD_CORPORA}/chime3_background" \
        > data/noise.scp
    log "data/noise.scp: $(wc -l < data/noise.scp) noise files"
    if [ ! -s data/rir.scp ]; then
        python3 ${v2}/rir_list.py "${VAD_CORPORA}" data/rir > data/rir.scp.tmp
        mv data/rir.scp.tmp data/rir.scp
    fi
    log "data/rir.scp: $(wc -l < data/rir.scp) impulse responses"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: corpora from their archives: ${corpora}"
    for corpus in ${corpora//,/ }; do
        if [ -f "${dumpdir}/${corpus}/.done" ]; then
            log "Skip ${corpus}: ${dumpdir}/${corpus}/.done exists"
            continue
        fi
        ${train_cmd} JOB=1:${nshards} "${dumpdir}/log/${corpus}.JOB.log" \
            python3 ${v2}/prep.py --corpus "${corpus}" --root "${VAD_CORPORA}" \
                --out "${dumpdir}" --shard "\$((JOB-1))" --nshards ${nshards} \
                --pack_seconds ${pack_seconds}
        touch "${dumpdir}/${corpus}/.done"
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: CTC alignment (${align_model}@${align_revision})"
    # the dump directories of --corpora with transcripts and no .aligned yet,
    # one GPU job each
    todo=()
    for corpus in ${corpora//,/ }; do
        [ -d "${dumpdir}/${corpus}" ] || continue
        while read -r d; do
            [ -f "${d}/.aligned" ] || todo+=("${d}")
        done < <(find "${dumpdir}/${corpus}" -name text -printf '%h\n' | sort)
    done
    for d in "${todo[@]}"; do
        name=$(echo "${d#"${dumpdir}"/}" | tr / _)
        ${cuda_cmd} --gpu 1 "${dumpdir}/log/align_${name}.log" \
            python3 ${v2}/align.py --dump "${d}" --model_tag "${align_model}" \
                --revision "${align_revision}" --device cuda \
                --min_pause "${align_min_pause}" \
                --min_word_score "${align_min_word_score}" &
    done
    wait
    for d in "${todo[@]}"; do
        [ -f "${d}/.aligned" ] || { log "${d} is not aligned: see ${dumpdir}/log"; exit 1; }
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: combine into dump/raw/{train,dev}, and the corpus reports"
    for split in train dev; do
        dirs=$(find "${dumpdir}" -mindepth 2 -maxdepth 2 -type d -name "${split}" | sort)
        # a transcribed dump must have been aligned
        for d in ${dirs}; do
            while read -r t; do
                [ -f "${t}/.aligned" ] || { log "${t} is not aligned: run stage 4"; exit 1; }
            done < <(find "${d}" -name text -printf '%h\n')
        done
        # shellcheck disable=SC2086
        python3 ${v2}/combine.py "dump/raw/${split}" ${dirs}
        # diar.sh skips stage 2 for a set marked done in dump/raw/org/<set>
        mkdir -p "dump/raw/org/${split}"
        for f in wav.scp utt2num_samples espnet_rttm feats_type utt2category; do
            cp "dump/raw/${split}/${f}" "dump/raw/org/${split}/${f}"
        done
        echo "omniio 16k" > "dump/raw/org/${split}/.stage2_done"
        log "dump/raw/${split}: $(wc -l < "dump/raw/${split}/wav.scp") recordings"
        column -t -s $'\t' "dump/raw/${split}/domains.tsv"
    done
    python3 ${v2}/corpus_report.py "${dumpdir}" "${dumpdir}/reports"
    log "corpus reports: ${dumpdir}/reports"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
