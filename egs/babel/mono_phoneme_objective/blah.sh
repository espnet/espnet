source ./conf/lang.conf
x=dev
#awk '(NR==FNR) {a[$1]=$0; next} ($1 in a){print $0}' data/${x}/text ${phoneme_ali} > data/${x}/text.phn
#./utils/filter_scp.pl data/${x}/text.phn data/${x}/text > data/${x}/text.tmp
mv data/${x}/text.tmp data/${x}/text
./utils/fix_data_dir.sh data/${x}
