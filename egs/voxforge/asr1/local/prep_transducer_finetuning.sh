#!/bin/bash

set -e

if [ "$#" -lt 4 ] || [ "$#" -gt 10 ]; then
    echo "Returns a set of configs to run pre-training(s) + transducer training with pre-initialization(s)"
    echo "Usage: $0 <main-conf-path> <transfer-type> <rnnt_mode> <backend>"
    echo "Accepted values: transfer_type: [enc, dec, both]"
    echo "                 rnnt_mode: [rnnt, rnnt-att]"
    echo "                 backend: [pytorch]"
    echo "Options: --output <output-name>"
    echo "         --enc-conf <encoder-configuration-path>"
    echo "         --enc-mods <encoder-module-list>"
    echo "         --enc-crit <encoder-model-criterion>"
    echo "         --dec-conf <decoder-configuration-path>"
    echo "         --dec-mods <decoder-module-list>"
    echo "         --dec-crit <decoder-model-criterion>"
    exit 1;
fi

main_conf=$1
transfer_type=$2
rnnt_mode=$3
backend=$4

[ ! -f "${main_conf}" ] && echo "Main config doesn't exit: $main_conf" && exit 1

case "$rnnt_mode" in
    rnnt|rnnt-att) ;;
    *) echo "Error: --rnnt-mode should be either rnnt or rnnt-att" && exit 1
esac

case "$transfer_type" in
    enc|dec|both) ;;
    *) echo "Error: --transfer-type should be either enc, dec or both" && exit 1
esac

[ "${backend}" != "pytorch" ] && echo "Error: only pytorch backend is supported" && exit 1

shift 4
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output)
            output=$2
            ;;
        --enc-conf)
            enc_conf=$2
            ;;
        --enc-mods)
            enc_mods="'$2'"
            ;;
        --enc-criterion)
            enc_crit=$2
            ;;
        --dec-conf)
            dec_conf=$2
            ;;
        --dec-mods)
            dec_mods="'$2'"
            ;;
        --dec-criterion)
            dec_crit=$2
    esac
    shift
done

[ -z "${output}" ] && output=$(dirname "$main_conf")/finetuning.yaml
rm -f ${output}

[[ "$transfer_type" =~ ^(enc|both)$ ]] && [ -z "${enc_conf}" ] && \
    enc_conf=conf/tuning/transducer/pretrain_ctc.yaml && create_enc=true

enc_is_transformer=$(grep -E "^etype:[[:space:]]*transformer" ${main_conf} || true)
dec_is_transformer=$(grep -E "^dtype:[[:space:]]*transformer" ${main_conf} || true)

if [ -n "${enc_is_transformer}" ]; then
    enc_mods=${enc_mods:-"'encoder.'"}
else
    enc_mods=${enc_mods:-"'enc.enc.'"}
fi
enc_crit=${enc_crit:-loss}

if [[ "$transfer_type" =~ ^(dec|both)$ ]] && [ -z "${dec_conf}" ]; then
    create_dec=true

    if [ "${rnnt_mode}" == "rnnt" ]; then
        dec_conf=conf/tuning/transducer/pretrain_lm.yaml
        [ -z "${dec_mods}" ] && dec_mods="'predictor.rnn.'"
        [ -z "${dec_crit}" ] && dec_crit=loss
    else
        dec_conf=conf/tuning/transducer/pretrain_att.yaml
        [ -z "${dec_mods}" ] && dec_mods="'dec.decoder.,dec.att.,att.'"
        [ -z "${dec_crit}" ] && dec_crit=acc
    fi
fi

exp_conf=$(dirname "$main_conf")/$(basename "${main_conf%.*}")_${transfer_type}_init.yaml
sed '$a\' ${main_conf} > ${exp_conf}

echo "exp-conf: ${exp_conf}" >> ${output}
case "${transfer_type}" in
    enc|both)
        echo "enc-conf: ${enc_conf}" >> ${output}
        printf '%s\n' \
               "enc-init: 'exp/tr_it_${backend}_$(basename ${enc_conf%.*})/results/model.${enc_crit}.best'" \
               "enc-init-mods: ${enc_mods}" >> ${exp_conf}
        ;;&
    dec|both)
        echo "dec-conf: ${dec_conf}" >> ${output}
        case "${rnnt_mode}" in
            rnnt)
                printf '%s\n' \
                       "dec-init: 'exp/train_rnnlm_${backend}_$(basename ${dec_conf%.*})/rnnlm.model.best'" \
                       "dec-init-mods: ${dec_mods}" >> ${exp_conf} ;;
            rnnt-att)
                printf '%s\n' \
                       "dec-init: 'exp/tr_it_${backend}_$(basename ${dec_conf%.*})/results/model.${dec_crit}.best'" \
                       "dec-init-mods: ${dec_mods}" >> ${exp_conf} ;;
        esac
        ;;
esac

sed -i -r "s/(^rnnt-mode:) ('[a-z-]*')/\1 '${rnnt_mode}'/g" ${exp_conf}

gen_section='/^# network/,/^($ | $)/d'
net_section='/^# network/,$p'

if [ "${create_enc}" == "true" ]; then
    rm -f ${enc_conf}

    use_vgg2l=$(grep -E "^transformer-input-layer:[[:space:]]*vgg2l" ${exp_conf} || true)

    if [ -n "${enc_is_transformer}" ]; then
        tt_section='1,/^# transducer/{/^# transducer/!p}'
    else
        tt_section='1,/^## attention/{/^## attention/!p}'
    fi

    sed "${gen_section}" ${exp_conf} |
        sed -e '/early-stop-criterion/d' -e '/criterion/d' >> ${enc_conf}

    sed -n -r "${net_section}" ${exp_conf} | \
        sed -n "${tt_section}" | \
        sed -e "/dropout/d" -e "/embed/d" >> ${enc_conf} && \
        echo -e "# CTC mode\nmtlalpha: 1.0" >> ${enc_conf}

    if [ -n "${enc_is_transformer}" ]; then
        sed -i -e "/etype/d" -e "/joint-dim/d" -e "/dtype/d" ${enc_conf} && \
            echo -e "model-module: \"espnet.nets.pytorch_backend.e2e_asr_transformer:E2E\"" \
                 >> ${enc_conf} && \
            echo -e "criterion: loss" >> ${enc_conf}

        if [ -n "${use_vgg2l}" ]; then
           echo "VGG2L is only available for transducer, thus it can't be used in transfer learning."
           echo "Switching to conv2d in main transducer conf and pre-training enc conf."

           sed -i "s/vgg2l/conv2d/g" ${exp_conf}
           sed -i "s/vgg2l/conv2d/g" ${enc_conf}
        fi
    fi
fi

if [ "${create_dec}" == "true" ]; then
    rm -rf ${dec_conf}

    if [ ${rnnt_mode} == "rnnt-att" ]; then
        sed "${gen_section}" ${exp_conf} | \
            sed '/early-stop-criterion/d' | \
            sed -r "s/([a-z]*:) loss/\1 ${dec_crit}/g" >> ${dec_conf}
        sed -n -r "${net_section}" ${exp_conf} | \
            sed -e "/dropout/d" -e "/embed/d" >> ${dec_conf} && \
            echo -e "\n# Att mode\nmtlalpha: 0.0" >> ${dec_conf}
    elif [ ${rnnt_mode} == "rnnt" ]; then
        dflt="opt: adam\nbatchsize: 512\nepoch: 20\npatience: 3\nmaxlen: 150\n"

        echo -e "$dflt" >> ${dec_conf}
        grep -E "(^dunits:|^dlayers:)" ${exp_conf} | \
            sed -r "s/^d(unit|layer)s: ([0-9]*)/\1: \2/g" >> ${dec_conf}
    fi
fi
