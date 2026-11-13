#!/usr/bin/env bash
# Prepare the training and validation dumps of frontia_vad_v2, and the test sets.
#
# The training corpora are written straight from their archives under ${VAD_CORPORA}
# (db.sh) into 16 kHz Omni-IO dumps, one per corpus and split, with labels and
# domains (local/v2/vad_dump.py); nothing is unpacked. Then:
#   dump/raw/train, dump/raw/dev   all training corpora, combined
#   data/<corpus>_{dev,eval}        test sets of frontia_vad_v1 (dumped by diar.sh)
#
# Stages:
#   1  meeting corpora of v1 (AMI, ICSI, CHiME-6, NOTSOFAR-1) and the v1 test
#      sets: Kaldi directories by the v1 scripts (local/data_v1.sh), then the
#      training splits into v2 dumps (local/v2/from_kaldi.py)
#   2  every corpus in --corpora (local/v2/adapters*.py), --nshards jobs each
#   3  CTC forced alignment of the transcribed corpora (label source "A") with
#      OWSM-CTC v4 at --align_revision, on GPU (local/v2/align.py)
#   4  combine into dump/raw/{train,dev} (local/v2/combine.py)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=4
corpora=""          # comma separated adapter names; empty: adapters.training_corpora()
v1_train="ami,chime6,notsofar,icsi"
v1_test="dipco,ami_mh,alimeeting,aishell4,voxconverse,earnings21,ava_speech,ten_vad"
nshards=8           # jobs per corpus in stage 2
v1_nshards=8        # jobs per v1 corpus split in stage 1
align_model=espnet/owsm_ctc_v4_1B
align_revision=6075bf3d42e6037d1ef7abd7c5558ade886e6eaa
align_min_pause=0.2
align_min_word_score=-10.0
dumpdir=dump/v2

log "$0 $*"
. utils/parse_options.sh
. ./path.sh
. ./db.sh
. ./cmd.sh

[ -n "${VAD_CORPORA:-}" ] || { log "Set VAD_CORPORA (the root of the frontia_vad_v2 training corpora) in db.sh"; exit 1; }
v2=local/v2

if [ -z "${corpora}" ]; then
    corpora=$(python3 -c "import sys; sys.path.insert(0, '${v2}'); \
import adapters; print(','.join(adapters.training_corpora()))")
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: v1 meeting corpora and test sets"
    local/data_v1.sh --corpora "${v1_train},${v1_test}" --splits train,dev,eval
    # noise for augmentation: licence-filtered, no vocals, no babble (replaces
    # the list local/data_v1.sh wrote, which had all of MUSAN music and CHiME-3)
    python3 ${v2}/noise_list.py "${MUSAN}" "${CHIME3_BACKGROUND}" > data/noise.scp
    log "data/noise.scp: $(wc -l < data/noise.scp) noise files"
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
    log "Stage 2: corpora from their archives: ${corpora}"
    for corpus in ${corpora//,/ }; do
        if [ -f "${dumpdir}/${corpus}/.done" ]; then
            log "Skip ${corpus}: ${dumpdir}/${corpus}/.done exists"
            continue
        fi
        ${train_cmd} JOB=1:${nshards} "${dumpdir}/log/${corpus}.JOB.log" \
            python3 ${v2}/prep.py --corpus "${corpus}" --root "${VAD_CORPORA}" \
                --out "${dumpdir}" --shard "\$((JOB-1))" --nshards ${nshards}
        touch "${dumpdir}/${corpus}/.done"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: CTC alignment (${align_model}@${align_revision})"
    # every dump directory with transcripts and no .aligned yet, one GPU job each
    mapfile -t todo < <(find "${dumpdir}" -name text -printf '%h\n' | sort)
    for d in "${todo[@]}"; do
        [ -f "${d}/.aligned" ] && continue
        name=$(echo "${d#"${dumpdir}"/}" | tr / _)
        ${cuda_cmd} --gpu 1 "${dumpdir}/log/align_${name}.log" \
            python3 ${v2}/align.py --dump "${d}" --model_tag "${align_model}" \
                --revision "${align_revision}" --device cuda \
                --min_pause "${align_min_pause}" \
                --min_word_score "${align_min_word_score}" &
    done
    wait
    if find "${dumpdir}" -name text -printf '%h\n' | while read -r d; do
            [ -f "${d}/.aligned" ] || echo "${d}"; done | grep -q .; then
        log "Some dumps are not aligned; see ${dumpdir}/log/align_*.log"
        exit 1
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: combine into dump/raw/{train,dev}"
    for split in train dev; do
        dirs=$(find "${dumpdir}" -mindepth 2 -maxdepth 2 -type d -name "${split}" | sort)
        # a transcribed dump must have been aligned
        for d in ${dirs}; do
            if find "${d}" -name text | grep -q . && \
                    find "${d}" -name text -printf '%h\n' | while read -r t; do
                        [ -f "${t}/.aligned" ] || echo "${t}"; done | grep -q .; then
                log "${d} has unaligned transcripts: run stage 3"
                exit 1
            fi
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
        column -t -s $'\t' "dump/raw/${split}/domains.tsv" | head -100
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
