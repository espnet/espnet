#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1

remove_archive=false

if [ "$1" == --remove-archive ]; then
    remove_archive=true
    shift
fi

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--remove-archive] <data-base> <lang>"
    echo "e.g.: $0 /n/rd11/corpora_8/MUSTC_v1.0/ de"
    echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
lang=$2

if [ ! -d "${data}" ]; then
    echo "$0: no such directory ${data}"
    exit 1;
fi

langs="de_es_fr_it_nl_pt_ro_ru"
if [ ! $(echo ${langs} | grep ${lang}) ]; then
    echo "$0: no such lang ${lang}"
    exit 1;
fi

if [ ${lang} = "de" ]; then
    url=https://drive.google.com/open?id=1Mf2il_VelDIJMSio0bq7I8M9fSs-X4Ie
elif [ ${lang} = "es" ]; then
    url=https://drive.google.com/open?id=14d2ttsuEUFXsxx-KRWJMsFhQGrYOJcpH
elif [ ${lang} = "fr" ]; then
    url=https://drive.google.com/open?id=1acIBqcPVX5QXXXV9u8_yDPtCgfsdEJDV
elif [ ${lang} = "it" ]; then
    url=https://drive.google.com/open?id=1qbK88SAKxqjMUybkMeIjrJWnNAZyE8V0
elif [ ${lang} = "nl" ]; then
    url=https://drive.google.com/open?id=11fNraDQs-LiODDxyV5ZW0Slf3XuDq5Cf
elif [ ${lang} = "pt" ]; then
    url=https://drive.google.com/open?id=1C5qK1FckA702nsYcXwmGdzlMmHg1F_ot
elif [ ${lang} = "ro" ]; then
    url=https://drive.google.com/open?id=1nbdYR5VqcTbLpOB-9cICKCgsLAs7fVzd
elif [ ${lang} = "ru" ]; then
    url=https://drive.google.com/open?id=1Z3hSiP7fsR3kf8fjQYzIa07jmw4KXNnw
fi

if [ -f ${data}/.complete_en_${lang} ]; then
    echo "$0: data was already successfully extracted, nothing to do."
    exit 0;
fi

if [ -f ${data}/MUSTC_v1.0_en-${lang}.tar.gz ]; then
    echo "${data}/MUSTC_v1.0_en-${lang}.tar.gz exists and appears to be complete."
fi

if [ ! -f ${data}/MUSTC_v1.0_en-${lang}.tar.gz ]; then
    if ! which wget >/dev/null; then
        echo "$0: wget is not installed."
        exit 1;
    fi
    echo "$0: downloading data from ${url}.  This may take some time, please be patient."
    download_from_google_drive.sh ${url} ${data} tar.gz || exit 1
fi

if ! tar -zxvf ${data}/MUSTC_v1.0_en-${lang}.tar.gz -d -C ${data}; then
    echo "$0: error un-tarring archive ${data}/MUSTC_v1.0_en-${lang}.tar.gz"
    exit 1;
fi

touch ${data}/.complete_en_${lang}
echo "$0: Successfully downloaded and un-tarred ${data}/MUSTC_v1.0_en-${lang}.tar.gz"

if $remove_archive; then
    echo "$0: removing ${data}/MUSTC_v1.0_en-${lang}.tar.gz file since --remove-archive option was supplied."
    rm ${data}/MUSTC_v1.0_en-${lang}.tar.gz
fi
