#!/bin/bash
targetcase=lower #or upper
for i in $*; do
    case "$1" in
	-intact-wlist  | --intact-wlist)
	    intact_wlist=$2
            shift
            shift
	    ;;
        -locale | --locale)
            export LANG=$2
	    export LC_ALL=$LANG
            shift
            shift
            ;;
	-case | --case)
	    targetcase=$2
	    shift
	    shift
	    ;;
	-help | --help | -?)
	    echo "Usage: $0 [--intact-wlist wlist] [--locale locale] [--case lower/upper] [input/-] [output/-]"
	    echo " Convert all characters after first column into desired case"
	    echo " By default, utt2spk is expected to be sorted by both, which can be "
	    echo " achieved by making the speaker-id prefixes of the utterance-ids"
	    echo "   --locale locale      [default given by system]     Locale for current conversion"
	    echo "   --case upper/lower   [default lower]               Desired case"
	    echo "   --intact-wlist wlist [default None]                Given words will stay intact"
	    echo "e.g.: $0 data/train/text |"
	    exit 1;
	    shift
	    ;;
        *)
            break
            ;;
    esac
done

IN=$1
OU=$2

if [ "$IN" == "-" ] || [ -z $IN ]; then IN=/dev/stdin; fi
if [ "$OU" == "-" ] || [ -z $OU ]; then OU=/dev/stdout;fi

if [ "$targetcase" != "lower" ] && [ "$targetcase" != "upper" ]; then
    echo "ERROR: $0: Target case could be lower/upper; $targetcase found instead"
    exit 1
fi


awk '
k==0{WLIST[$1]=1}
k==1{
 for(i=2;i<=NF;i++) if(!($i in WLIST)) $i=to'$targetcase'($i)  
 print $0
 }' k=0 $intact_wlist k=1 $IN > $OU
