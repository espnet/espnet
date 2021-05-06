. utils/parse_options.sh

if [ $# != 5 ]; then
    echo "Usage: $0 <src-data-dir> <dest-trdata-dir> <dest-dtdata-dir> <dest-etdata-dir> <spk_list_conf>>";
    exit 1;
fi

sdata=$1
trdata=$2
dtdata=$3
etdata=$4
spk_list=$5

# get a temp dir
./utils/copy_data_dir.sh data/${sdata} data/${trdata}
./utils/copy_data_dir.sh data/${sdata} data/${dtdata}
./utils/copy_data_dir.sh data/${sdata} data/${etdata}

python3 ./local/split_tr_dt_et.py -s ${trdata} -c ${spk_list} --train ${trdata} --test ${etdata} --dev ${dtdata}
mv data/${trdata}/new_segments data/${trdata}/segments
mv data/${etdata}/new_segments data/${etdata}/segments
mv data/${dtdata}/new_segments data/${dtdata}/segments

./utils/fix_data_dir.sh data/${trdata}
./utils/fix_data_dir.sh data/${etdata}
./utils/fix_data_dir.sh data/${dtdata}
