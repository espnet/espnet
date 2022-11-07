#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0 <in-xml.scp> <out-datadir> [<logdir> [<outdir>]]
e.g.
$0 data/test/xml.scp data/test_format/

Format 'xml.scp': In short words,
changing "kaldi-datadir" to "modified-kaldi-datadir"

The 'xml.scp' format in kaldi is very flexible,
e.g. It can use unix-pipe as describing that xml file,
but it sometime looks confusing and make scripts more complex.
This tools creates actual xml files from 'xml.scp'.

Options
  --nj <nj>
  --segments <segments>
  --cmd <cmd>
EOF
)

out_xmlfilename=xml.scp
cmd=utils/run.pl
nj=30
segments=

ref_channels=
utt2ref_channels=

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 2 ] && [ $# -ne 3 ] && [ $# -ne 4 ]; then
    log "${help_message}"
    log "Error: invalid command line arguments"
    exit 1
fi

. ./path.sh  # Setup the environment

scp_dir=$1
# NOTE(Yuning): If orgin xmls are segmented here, scp_filw need to be fixed like wav.scp.
dir=$2


if [ $# -eq 2 ]; then
    logdir=${dir}/logs
    outdir=${dir}/data

elif [ $# -eq 3 ]; then
    logdir=$3
    outdir=${dir}/data

elif [ $# -eq 4 ]; then
    logdir=$3
    outdir=$4
fi


mkdir -p ${logdir}

rm -f "${dir}/${out_xmlfilename}"

opts=
if [ -n "${utt2ref_channels}" ]; then
    opts="--utt2ref-channels ${utt2ref_channels} "
elif [ -n "${ref_channels}" ]; then
    opts="--ref-channels ${ref_channels} "
fi


if [ -n "${segments}" ]; then
    log "[info]: using ${segments}"
    nutt=$(<${segments} wc -l)
    nj=$((nj<nutt?nj:nutt))
    
    ${cmd} "JOB=1:${nj}" "${logdir}/format_xml_scp.JOB.log" \
        pyscripts/audio/format_xml_scp.py \
            ${opts} \
            "--segment=${logdir}/segments.JOB" \
            "${scp_dir}" "${outdir}/format_xml.JOB"

else
    # TODO(Yuning): xml_scp without segments needs to be finished
    log "[info]: without segments"
    nutt=$(<${scp_dir}/xml.scp wc -l)
    nj=$((nj<nutt?nj:nutt))

    split_scps=""
    for n in $(seq ${nj}); do
        split_scps="${split_scps} ${logdir}/xml.${n}.scp"
    done

    utils/split_scp.pl "${scp_dir}/xml.scp" ${split_scps}
    
    ${cmd} "JOB=1:${nj}" "${logdir}/format_xml_scp.JOB.log" \
        pyscripts/audio/format_xml_scp.py \
        ${opts} \
        "--segment=${logdir}/segments.JOB" \
        "${scp_dir}/xml.scp" "${outdir}/format_xml.JOB"
fi

# Workaround for the NFS problem
ls ${outdir}/format_xml.* > /dev/null

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat "${outdir}/format_xml.${n}/xml.scp" || exit 1;
done > "${dir}/${out_xmlfilename}" || exit 1

log "Successfully finished. [elapsed=${SECONDS}s]"

