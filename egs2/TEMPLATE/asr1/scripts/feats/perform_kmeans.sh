#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
skip_stages=
cpu_cmd="run.pl"
num_threads=20      # number of cpu threads in learn_kmeans
percent=-1          # fraction of utterances to fit k-means on; -1 for all.
# MiniBatchKMeans controls, forwarded to learn_kmeans.py. Defaults below match
# that script's own defaults, so nothing changes unless you set them.
#
# WATCH max_iter: it is the number of full passes over every selected frame, and
# total steps = max_iter * n_frames / batch_size. With tol=0 the tol-based
# convergence test is disabled, and max_no_improvement=100 will not fire while
# the ewa inertia is still creeping down, so a run really does take all of them.
# Measured on 26.9 M 39-dim frames: 267,409 steps at 4.5 steps/s = 16 h, while
# the inertia was already within 1.7% of its asymptote after 0.8 of one pass.
# Per-step cost scales with batch_size * dim * nclusters, so at 1024 dims and
# 500 clusters the same defaults project to ~45 days.
# Fast paths for finalising a very large label file. All default to the
# original behaviour. Measured on 62.5 M utterances / a 148 GB label file, where
# finalising cost 285 min against 227 min of actual labelling compute.
assume_sorted_splits=false  # sort each per-job label file on its own instead of
                            # doing one external sort over the concatenation
                            # (measured 70 min on 149 GB -> ~20 min, and no
                            # 149 GB of sort temp space).
                            #
                            # Do NOT mistake the original `sort -u` for a
                            # no-op. dump_km_label reads through
                            # NumElementsBatchSampler, which orders utterances
                            # by LENGTH to form batches, so each per-job file
                            # comes out in length order, not utt-id order. The
                            # sort is what puts it back into id order; the `-u`
                            # is incidental (the splits are disjoint).
                            #
                            # This fast path is valid only because each split is
                            # a contiguous, increasing id range, so sorting the
                            # splits individually and concatenating them in job
                            # order yields a globally sorted file. Each split
                            # fits in memory, which is why it beats one external
                            # merge. The check below verifies BOTH properties:
                            # each split internally sorted, and each split
                            # ending before the next begins. An earlier version
                            # checked only the boundaries, which passes
                            # trivially for contiguous splits and missed the
                            # length ordering entirely.
fast_label_finalize=false   # hard-link the label file into the data dir instead
                            # of copying it, and skip fix_data_dir on it
                            # (67 min -> ~0, plus 148 GB of disk). fix_data_dir
                            # copies the file to .backup/ and cmp's it against a
                            # filtered copy: ~450 GB of I/O to conclude nothing
                            # needs filtering, which cannot happen because
                            # dump_km_label emits exactly one line per input
                            # utterance and the inputs partition wav.scp. The
                            # two halves are coupled -- the hard link is only
                            # safe because nothing then rewrites in place.
token_count_sample_utts=0   # >0 counts tokens from only this many utterances
                            # per per-job label file, instead of every token in
                            # the full file (88 min -> ~1 min). The counts only
                            # order a ~100-line vocabulary. Sampling per job is
                            # stratified across corpora; sampling the
                            # concatenated file would not be, because it is
                            # grouped by language. Must be a LINE count, not a
                            # byte count: `head -c` truncates the last line, and
                            # the fragment then fuses with the next file's first
                            # line, injecting utterance ids into the vocabulary.
split_by_duration=false   # split label-dumping jobs by total audio rather than
                         # by utterance count. split_scp.pl gives every job an
                         # equal NUMBER of utterances, which on a corpus of
                         # mixed-length material leaves jobs with wildly
                         # different amounts of audio: on 62.5 M utterances
                         # across 34 corpus/language groups, sorted utt ids put
                         # each split inside one corpus, and the heaviest split
                         # held 3897 h against the lightest 374 h -- 10.4x. The
                         # stage then runs at the pace of the worst straggler.
                         # Cutting on cumulative samples instead keeps every
                         # split contiguous and sorted, just with a variable
                         # utterance count.
km_max_iter=100
km_batch_size=10000
km_tol=0.0
km_max_no_improvement=100
km_n_init=20
                    # learn_kmeans.py holds every selected frame in RAM at once
                    # (load_feature_shard concatenates them), so for a
                    # high-dimensional SSL feature this is the memory knob:
                    # 1024-dim features over 299 h is 220 GB at -1.
cuda_cmd="run.pl"
nj=16               # number of parallel jobs
python=python3      # Specify python to execute espnet commands.
train_set=          # Name of training set
dev_set=            # Name of valid set
other_sets=         # Name of other sets
datadir=dump/raw    # Directory for the source speech data used to dump feature and label.
featdir=dump/hubert_feats   # Directory for the dumped features and labels.
km_dir=             # Directory for the kmeans models
dictdir=            # Directory for the fairseq dictionary (only used for hubert training)
alignment_phoneme_dir="data/mfa_phoneme_alignment"  # Directory for alignment labels
phn_sets="dev-other dev-clean"      # Datasets of alignment used to measure the pseudo-label quality
upsample=           # Upsampling rate of pseudo-labels to measure the pseudo-lable quality
use_gpu=false       # Whether to use gpu in feature extraction
suffix=             # A suffix to distinguish the feature dump directory. Empty in usual cases.
audio_format="wav"  # The audio format of the source speech (flac, wav, *_ark, etc)
audio_sample_rate=16000 # the sample rate of input audio

skip_train_kmeans=false     # Whether to skip the kmeans model training
nclusters=100       # Number of clusters of kmeans model
portion=0.1         # Portion of data from training set used to train kmeans model
storage_save_mode=false     # Save storage on SSL feature extraction
                            # If true, feature extraction and kmeans clustering on the fly

RVQ_layers=1

feature_conf=       # feature configuration in json string format
feature_type=mfcc   # mfcc / fairseq_hubert / espnet_hubert / espnet_wavlm
layer=              # The layer index of SSL models to extract features from.
batch_bins=         # batch size when extracting features and labels.

# Legacy Fairseq HuBERT model and ESPnet-trained HuBERT/WavLM model related for
# feature extraction.
# Example of legacy Fairseq HuBERT model
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models/hubert_base_ls960.pt"
# Example of espnet-trained model
# hubert_url="espnet"
# hubert_dir_path="" # Pretrained Hubert/WavLM model dir contains 'valid.acc.best.pth' and 'config.yaml'

log "$0 $*"
. utils/parse_options.sh

. ./path.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 <--nclusters:100> <--feature_type:mfcc>"
    exit 0
fi

if [ ${feature_type} = "mfcc" ]; then  # MFCC has no layer
    use_gpu=false
elif [ -z "${suffix}" ]; then
    suffix="layer${layer}/"
fi
if [ -z "${feature_conf}" ]; then
    feature_conf="{type=${feature_type}"
    if [ ${feature_type} = "espnet_hubert" ]; then
        feature_conf+=",conf={\
sample_rate=16000,hubert_model_path=${hubert_dir_path},\
layer=${layer}\
}"
    elif [ ${feature_type} = "espnet_wavlm" ]; then
        feature_conf+=",conf={\
sample_rate=16000,wavlm_model_path=${hubert_dir_path},\
layer=${layer}\
}"
    elif [ ${feature_type} = "fairseq_hubert" ]; then
        feature_conf+=",conf={\
sample_rate=16000,hubert_url=${hubert_url},\
hubert_dir_path=${hubert_dir_path},layer=${layer}\
}"
    elif [ ${feature_type} != "mfcc" ]; then
        log "Error: unsupported feature type ${feature_type}" && exit 2
    fi
    feature_conf+="}"
fi

if "${skip_train_kmeans}"; then
    skip_stages+=" 2"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "stage 1: Dump ${feature_type} feature"

    if ${use_gpu}; then
        _cmd="${cuda_cmd} --gpu 1"
    else
        _cmd="${cpu_cmd}"
    fi

    if [[ "${audio_format}" == *ark* ]]; then
        _in_filetype="kaldi_ark"
    else
        # "sound" supports "wav", "flac", etc.
        _in_filetype="sound"
    fi

    if ${storage_save_mode}; then
        _dsets="${train_set}_subset${portion}"
        mkdir -p "${datadir}/${_dsets}"

        nutt=$(<"${datadir}/${train_set}"/wav.scp wc -l)
        portion_nutt=$(echo ${nutt} ${portion} | awk '{print(int($1 * $2))}')
        portion_nutt=$(( portion_nutt > 0 ? portion_nutt : 1 ))

        utils/subset_data_dir.sh \
            "${datadir}/${train_set}" ${portion_nutt} "${datadir}/${_dsets}"
        utils/filter_scp.pl ${datadir}/${_dsets}/utt2spk \
            <${datadir}/${train_set}/utt2num_samples >${datadir}/${_dsets}/utt2num_samples
        log "Subsampling ${portion_nutt} utterances for feature dumping."
    else
        _dsets="${train_set} ${other_sets} ${dev_set}"
    fi
    for dset in ${_dsets}; do
        echo "Dump SSL ${dset} features to ${featdir}/${feature_type}/${suffix}${dset}"
        _dump_dir="${featdir}/${feature_type}/${suffix}${dset}"

        utils/copy_data_dir.sh --validate_opts --non-print "${datadir}/${dset}" "${_dump_dir}"

        # 1. Split the key file
        output_dir="${_dump_dir}/data"
        mkdir -p "${output_dir}"
        _logdir="${_dump_dir}/logdir"
        mkdir -p "${_logdir}"

        nutt=$(<"${_dump_dir}"/wav.scp wc -l)
        _nj=$((nj<nutt?nj:nutt))

        key_file="${datadir}/${dset}"/wav.scp
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/wav.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        for n in $(seq ${_nj}); do
            awk '(FILENAME==ARGV[1]){utt2num[$1]=$2} (FILENAME==ARGV[2]){print($1, utt2num[$1])}' \
                ${datadir}/${dset}/utt2num_samples ${_logdir}/wav.${n}.scp > ${_logdir}/utt2num_samples.${n}
        done

        # shellcheck disable=SC2046,SC2086
        ${_cmd} JOB=1:${_nj} ${_logdir}/dump_features.JOB.log \
            ${python} pyscripts/feats/dump_ssl_feature.py \
                --feature_conf "'${feature_conf}'" \
                --audio_sample_rate "${audio_sample_rate}" \
                --use_gpu ${use_gpu} \
                --in_filetype "${_in_filetype}" \
                --out_filetype "mat" \
                --write_num_frames "ark,t:${output_dir}/utt2num_frames.JOB" \
                --utt2num_samples "${_logdir}/utt2num_samples.JOB" \
                ${batch_bins:+--batch_bins ${batch_bins}} \
                "scp:${_logdir}/wav.JOB.scp" \
                "ark,scp:${output_dir}/feats.JOB.ark,${output_dir}/feats.JOB.scp" || exit 1;

        # concatenate scp files
        for n in $(seq ${_nj}); do
            cat ${output_dir}/feats.${n}.scp || exit 1;
        done > ${output_dir}/../feats.scp || exit 1

        for n in $(seq ${_nj}); do
            cat ${output_dir}/utt2num_frames.$n || exit 1;
        done > ${output_dir}/../utt2num_frames || exit 1
    done

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    log "stage 2: Learn K-means with ${feature_type} feature based on scikit-learn"

    _logdir="${km_dir}/logdir"
    mkdir -p ${_logdir}

    if ${storage_save_mode}; then
        _portion=1.0
        _dset="${train_set}_subset${portion}"
    else
        _portion=${portion}
        _dset="${train_set}"
    fi

    # select portion of data
    if (( $(echo "${_portion} >= 1.0" | bc -l) )); then
        cp "${featdir}/${feature_type}/${suffix}${_dset}"/feats.scp "${km_dir}/train.scp"
    else
        nutt=$(<"${featdir}/${feature_type}/${suffix}${_dset}"/feats.scp wc -l)
        portion_nutt=$(echo ${nutt} ${_portion} | awk '{print(int($1 * $2)+1)}')

        subset_scp.pl \
            ${portion_nutt} ${featdir}/${feature_type}/${suffix}${_dset}/feats.scp \
            > "${km_dir}/train.scp" || exit 1;
        log "Subsampling ${portion_nutt} utterances for Kmeans training."
    fi

    # It typically requires 120GB RAM to run kmeans steps.
    ${cpu_cmd} --num_threads ${num_threads} ${_logdir}/learn_kmeans.log \
        ${python} pyscripts/utils/learn_kmeans.py \
            --km_path ${km_dir}/km_${nclusters}.mdl \
            --n_clusters ${nclusters} \
            --RVQ_layers ${RVQ_layers} \
            --percent ${percent} \
            --max_iter ${km_max_iter} \
            --batch_size ${km_batch_size} \
            --tol ${km_tol} \
            --max_no_improvement ${km_max_no_improvement} \
            --n_init ${km_n_init} \
            --in_filetype mat \
            "scp:${km_dir}/train.scp" || exit 1;
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    log "stage 3: Generate K-means pseudo-labels"

    if ${use_gpu}; then
        _cmd="${cuda_cmd} --gpu 1"
    else
        _cmd="${cpu_cmd}"
    fi

    for dset in "${train_set}" "${dev_set}" ${other_sets}; do
        log "Extract labels to ${featdir}/${feature_type}/${suffix}${dset}"

        _dump_dir="${featdir}/${feature_type}/${suffix}${dset}"

        _opts=
        if ${storage_save_mode}; then
            utils/copy_data_dir.sh --validate_opts --non-print "${datadir}/${dset}" "${_dump_dir}"
            key="wav.scp"
            if [[ "${audio_format}" == *ark* ]]; then
                _opts+="--in_filetype kaldi_ark "
            else
                # "sound" supports "wav", "flac", etc.
                _opts+="--in_filetype sound "
            fi
            _opts+="--online_feature_extract ${storage_save_mode} "
            _opts+="--feature_conf \"${feature_conf}\" "
            if [ -n "${batch_bins}" ]; then
                _opts+="--batch_bins ${batch_bins} "
            fi
        else
            key="feats.scp"
            _opts+="--in_filetype mat "
        fi
        mkdir -p "${_dump_dir}"/logdir

        nutt=$(<"${_dump_dir}"/${key} wc -l)
        _nj=$((nj<nutt?nj:nutt))

        key_file="${_dump_dir}"/${key}
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_dump_dir}/logdir/inference_kmeans.${n}.scp"
        done
        if ${split_by_duration}; then
            # One pass writes both the wav.scp splits and their utt2num_samples
            # companions, cutting whenever cumulative audio crosses the next
            # 1/_nj boundary. Splits stay contiguous and sorted, so everything
            # downstream is unchanged -- only the boundaries move.
            awk -v dir="${_dump_dir}/logdir" -v nj="${_nj}" '
                NR==FNR { n[$1]=$2; total+=$2; next }
                FNR==1  { target = total / nj; k = 1
                          scp = dir "/inference_kmeans." k ".scp"
                          u2n = dir "/utt2num_samples." k }
                {
                  print           > scp
                  print $1, n[$1] > u2n
                  acc += n[$1]
                  if (k < nj && acc >= k * target) {
                      close(scp); close(u2n); k++
                      scp = dir "/inference_kmeans." k ".scp"
                      u2n = dir "/utt2num_samples." k
                  }
                }' ${datadir}/${dset}/utt2num_samples "${key_file}"
        else
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # One pass, not one per job. This used to be a `for n in $(seq ${_nj})`
        # loop whose awk re-read the whole utt2num_samples and rebuilt a
        # full-corpus hash every iteration: at 62.5 M utterances and nj=128 that
        # is 128 x 3.7 GB = 478 GB of single-threaded reads, measured at
        # 1.3 splits/min (~90 min), and it got *worse* as nj went up -- exactly
        # backwards. Reading the big file once and streaming each split to its
        # own output is identical in result and ~100x cheaper. Peak memory is
        # unchanged (the same single hash). Output name is derived from
        # FILENAME, so it does not depend on argument order, and each output is
        # closed before the next opens so awk holds only one at a time.
        # shellcheck disable=SC2086
        awk 'NR==FNR { utt2num[$1]=$2; next }
             FNR==1 { if (out != "") close(out)
                      out = FILENAME
                      sub(/inference_kmeans\./, "utt2num_samples.", out)
                      sub(/\.scp$/, "", out) }
             { print $1, utt2num[$1] > out }' \
            ${datadir}/${dset}/utt2num_samples ${split_scps}
        fi

        ${_cmd} JOB=1:${_nj} "${_dump_dir}"/logdir/inference_pseudo_labels_km${nclusters}.JOB.log \
            ${python} pyscripts/feats/dump_km_label.py \
                ${_opts} \
                --audio_sample_rate "${audio_sample_rate}" \
                --km_path "${km_dir}/km_${nclusters}.mdl" \
                --RVQ_layers "${RVQ_layers}" \
                --out_filetype "mat" \
                --use_gpu ${use_gpu} \
                --utt2num_samples "${_dump_dir}/logdir/utt2num_samples.JOB" \
                "scp:${_dump_dir}/logdir/inference_kmeans.JOB.scp" \
                "ark,t:${_dump_dir}/logdir/pseudo_labels_km${nclusters}.JOB.txt" || exit 1;

        # concatenate scp files
        for layer_idx in $(seq 1 $((RVQ_layers))); do
            tail_="km${nclusters}"
            if [ ${RVQ_layers} -gt 1 ]; then
                tail_="RVQ_$((layer_idx-1))_km${nclusters}"
            fi
            if ${assume_sorted_splits}; then
                # Verify the id ranges do not interleave, then sort each split
                # on its own. Sorting per split is what makes this cheap; the
                # ranges being disjoint and increasing is what makes it correct.
                _prev=
                for n in $(seq ${_nj}); do
                    _f="${_dump_dir}/logdir/pseudo_labels_${tail_}.${n}.txt"
                    # One awk, no pipeline: `... | sort | head -1` makes sort
                    # take SIGPIPE, which under `set -o pipefail` aborts the
                    # whole script from inside a command substitution, i.e.
                    # silently. Also O(n) with no sort at all.
                    _range=$(awk 'NR==1{lo=hi=$1} {if($1<lo)lo=$1; if($1>hi)hi=$1} END{print lo, hi}' "${_f}")
                    _lo=${_range%% *}
                    _hi=${_range##* }
                    if [ -n "${_prev}" ] && ! [[ "${_prev}" < "${_lo}" ]]; then
                        log "Error: label split ${n} interleaves with the previous one" \
                            "(${_prev} !< ${_lo}); rerun without --assume_sorted_splits"
                        exit 1
                    fi
                    _prev="${_hi}"
                done
                log "split id ranges verified disjoint across ${_nj} files; sorting per split"
                for n in $(seq ${_nj}); do
                    LC_ALL=C sort "${_dump_dir}"/logdir/pseudo_labels_${tail_}.${n}.txt || exit 1;
                done | sed 's/ \[ \| \]//g' > "${_dump_dir}"/pseudo_labels_${tail_}.txt || exit 1;
            else
                for n in $(seq ${_nj}); do
                    cat "${_dump_dir}"/logdir/pseudo_labels_${tail_}.${n}.txt || exit 1;
                done | sed 's/ \[ \| \]//g' | sort -u > "${_dump_dir}"/pseudo_labels_${tail_}.txt || exit 1;
            fi
        done
    done
fi


km_tag=$(basename ${km_dir})

if [ -n "${alignment_phoneme_dir}" ]; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
        log "Stage 4: Measure qualities of pseudo labels"

        if [ -z "${upsample}" ]; then
            # upsample the pseudo labels to match the length of alignment
            if [ "${feature_type}" = "mfcc" ]; then
                upsample=1
            else
                upsample=2
            fi
        fi

        if [ -d "${alignment_phoneme_dir}" ]; then
            # TODO(simpleoier): This script and arguments design are specific to LibriSpeech dataset.
            ${python} local/measure_teacher_quality.py \
                --lab_dir "${featdir}/${feature_type}/${suffix}" \
                --lab_name "pseudo_labels_km${nclusters}.txt" \
                --lab_sets "${dev_set}" \
                --phn_dir "${alignment_phoneme_dir}" \
                --phn_sets ${phn_sets} \
                --pad_len 0 \
                --upsample ${upsample} \
                --ref_lab_dir "" \
                --ref_lab_name "" | tee ${km_dir}/phoneme_pseudo_label_quality.txt
        else
            log "Skipping quality measurement because no ${alignment_phoneme_dir} exists. You can specify the \
alignment by \"--alignment_phoneme_dir\". The alignment is in tsv file with format: \"utt_id1 a1,a2,a3,...\""
        fi
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    log "stage 5: Prepare pseudo-labels for training and dictionary: <token> <count>"

    for dset in "${train_set}" "${dev_set}" ${other_sets}; do
        label_dir="${featdir}/${feature_type}/${suffix}${dset}"
        if [ -f "${label_dir}"/pseudo_labels_km${nclusters}.txt ]; then
            if ${fast_label_finalize}; then
                # Hard link, not copy: instant and costs no extra disk. Safe
                # only because fix_data_dir is skipped below, so nothing
                # rewrites this file in place.
                rm -f ${datadir}/${dset}/text.km.${km_tag}
                ln "${label_dir}"/pseudo_labels_km${nclusters}.txt \
                   ${datadir}/${dset}/text.km.${km_tag} \
                  || cp "${label_dir}"/pseudo_labels_km${nclusters}.txt \
                        ${datadir}/${dset}/text.km.${km_tag}
            else
                cp "${label_dir}"/pseudo_labels_km${nclusters}.txt ${datadir}/${dset}/text.km.${km_tag}
            fi
        fi
        if ${fast_label_finalize}; then
            # Cheap end-to-end check in place of fix_data_dir: the label file
            # must start and end on the same utterances as wav.scp. This catches
            # truncation and misalignment without reading 450 GB.
            _w_first=$(head -1 ${datadir}/${dset}/wav.scp | cut -d' ' -f1)
            _w_last=$(tail -1 ${datadir}/${dset}/wav.scp | cut -d' ' -f1)
            _l_first=$(head -1 ${datadir}/${dset}/text.km.${km_tag} | cut -d' ' -f1)
            _l_last=$(tail -1 ${datadir}/${dset}/text.km.${km_tag} | cut -d' ' -f1)
            if [ "${_w_first}" != "${_l_first}" ] || [ "${_w_last}" != "${_l_last}" ]; then
                log "Error: text.km.${km_tag} does not span wav.scp for ${dset}" \
                    "(wav ${_w_first}..${_w_last} vs km ${_l_first}..${_l_last});" \
                    "rerun without --fast_label_finalize"
                exit 1
            fi
            log "${dset}: label file spans wav.scp (${_l_first}..${_l_last}); skipping fix_data_dir"
        else
            utils/fix_data_dir.sh --utt_extra_files "text.km.${km_tag}" ${datadir}/${dset}
        fi
    done

    # generate dictionaries
    if [ -n "${dictdir}" ]; then
        mkdir -p ${dictdir}

        oov="<unk>"         # Out of vocabulary symbol.
        blank="<blank>"     # CTC blank symbol
        pad="<pad>"
        sos_eos="<sos/eos>" # sos and eos symbole

        if [ "${token_count_sample_utts}" -gt 0 ]; then
            # Stratified sample: the first N MB of each per-job label file. The
            # counts only decide the ORDER of a ~100-line vocabulary, and every
            # job covers a different slice of the corpus, so this gives the same
            # ordering as a full pass over 148 GB. Sampling the concatenated
            # file with head would be biased -- it is grouped by language.
            _lab_dir="${featdir}/${feature_type}/${suffix}${train_set}/logdir"
            log "counting tokens from ${token_count_sample_utts} utts per job file (sampled)"
            for _f in "${_lab_dir}"/pseudo_labels_km${nclusters}.*.txt; do
                head -n "${token_count_sample_utts}" "${_f}"
            done | sed 's/ \[ \| \]//g' | cut -d" " -f2- | \
                awk '{for (i=1; i<=NF; i++) {count[$i]+=1}} END{for (k in count) {print(k, count[k])}}' | \
                    sort -n -r -k 2  | \
                    awk -v oov=${oov} -v blank=${blank} -v sos_eos=${sos_eos} -v pad=${pad} \
                        '{print($1)} END{print(oov); print(sos_eos)}' \
                    > ${dictdir}/tokens.txt
        else
        <${datadir}/${train_set}/text.km.${km_tag} cut -d" " -f2- | \
            awk '{for (i=1; i<=NF; i++) {count[$i]+=1}} END{for (k in count) {print(k, count[k])}}' | \
                sort -n -r -k 2  | \
                awk -v oov=${oov} -v blank=${blank} -v sos_eos=${sos_eos} -v pad=${pad} \
                    '{print($1)} END{print(oov); print(sos_eos)}' \
                > ${dictdir}/tokens.txt
        fi

        log "Successfully generate the ${dictdir}/{dict,tokens}.txt"
    fi

fi
