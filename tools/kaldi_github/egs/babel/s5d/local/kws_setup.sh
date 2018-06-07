#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

# Begin configuration section.
cmd=run.pl
case_insensitive=true
subset_ecf=
rttm_file=
extraid=
use_icu=true
icu_transform="Any-Lower"
kwlist_wordlist=false
langid=107
annotate=true
silence_word=  # Optional silence word to insert (once) between words of the transcript.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

set -e
set -u
set -o pipefail

help_message="$0: Initialize and setup the KWS task directory
Usage:
       $0  <ecf_file> <kwlist-file> [rttm-file] <lang-dir> <data-dir>
allowed switches:
      --subset-ecf /path/to/filelist     # The script will subset the ecf file
                                         # to contain only the files from the filelist
      --rttm-file /path/to/rttm          # the preferred way how to specify the rttm
                                         # the older way (as an in-line parameter is
                                         # obsolete and will be removed in near future
      --case-insensitive <true|false>      # Shall we be case-sensitive or not?
                                         # Please not the case-sensitivness depends
                                         # on the shell locale!
      --annotate <true|false>
      --use-icu <true|false>           # Use the ICU uconv binary to normalize casing
      --icu-transform <string>           # When using ICU, use this transliteration
      --kwlist-wordlist                  # The file with the list of words is not an xml
              "

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

if [ "$#" -ne "5" ] &&  [ "$#" -ne "4" ] ; then
    printf "FATAL: invalid number of arguments.\n\n"
    printf "$help_message\n"
    exit 1
fi

ecf_file=$1
kwlist_file=$2
if [ "$#" -eq "5" ] ; then
    rttm_file=$3
    langdir=$4
    datadir=$5
else
    langdir=$3
    datadir=$4
fi

# don't quote rttm_file as it's valid for it to be empty.
for filename in "$ecf_file" "$kwlist_file" $rttm_file; do
    echo $filename
    if [ ! -e $filename ] ; then
        printf "FATAL: filename \'$filename\' does not refer to a valid file\n"
        printf "$help_message\n"
        exit 1;
    fi
done
for dirname in "$langdir" "$datadir" ; do
    if [ ! -d $dirname ] ; then
        printf "FATAL: dirname \'$dirname\' does not refer to a valid directory\n"
        printf "$help_message\n"
        exit 1;
    fi
done

if [ ! -z $extraid ]; then
  kwsdatadir=$datadir/${extraid}_kws
else
  kwsdatadir=$datadir/kws
fi

mkdir -p $kwsdatadir

if [ -z $subset_ecf ] ; then
  test -f $kwsdatadir/ecf.xml && rm -f $kwsdatadir/ecf.xml
  cp "$ecf_file" $kwsdatadir/ecf.xml || exit 1
else
  local/make_ecf_subset.sh $subset_ecf $ecf_file > $kwsdatadir/ecf.xml
fi

if $kwlist_wordlist ; then
(
 echo '<kwlist ecf_filename="kwlist.xml" language="" encoding="UTF-8" compareNormalize="lowercase" version="" >'
 awk '{ printf("  <kw kwid=\"%s\">\n", $1);
        printf("    <kwtext>"); for (n=2;n<=NF;n++){ printf("%s", $n); if(n<NF){printf(" ");} }
        printf("</kwtext>\n");
        printf("  </kw>\n"); }' < ${kwlist_file}
 # while read line; do
 #   id_str=`echo $line | cut -f 1 -d ' '`
 #   kw_str=`echo $line | cut -f 2- -d ' '`
 #   echo "  <kw kwid=\"$id_str\">"
 #   echo "    <kwtext>$kw_str</kwtext>"
 #   echo "  </kw>"
 # done < ${kwlist_file}
 echo '</kwlist>'
) > $kwsdatadir/kwlist.xml || exit 1
else
  test -f $kwsdatadir/kwlist.xml && rm -f $kwsdatadir/kwlist.xml
  cp "$kwlist_file"  $kwsdatadir/kwlist.xml || exit 1
fi

if [ ! -z $rttm_file ] ; then
  test -f $kwsdatadir/rttm && rm -f $kwsdatadir/rttm
  cp "$rttm_file" $kwsdatadir/rttm || exit 1
fi

sil_opt=
[ ! -z $silence_word ] && sil_opt="--silence-word $silence_word"
local/kws_data_prep.sh --case-insensitive ${case_insensitive} \
  $sil_opt --use_icu ${use_icu} --icu-transform "${icu_transform}" \
  $langdir $datadir $kwsdatadir || exit 1

if  $annotate ; then
  set -x
  rm -f $kwsdatadir/kwlist.xml
  cat $kwsdatadir/keywords.txt | local/search/create_categories.pl | local/search/normalize_categories.pl > $kwsdatadir/categories
  cat "$kwlist_file" | local/search/annotate_kwlist.pl $kwsdatadir/categories > $kwsdatadir/kwlist.xml || exit 1
fi
#~  (
#~  echo '<kwlist ecf_filename="kwlist.xml" language="" encoding="UTF-8" compareNormalize="lowercase" version="" >'
#~  while read line; do
#~    id_str=`echo $line | cut -f 1 -d ' '`
#~    kw_str=`echo $line | cut -f 2- -d ' '`
#~    echo "  <kw kwid=\"$id_str\">"
#~    echo "    <kwtext>$kw_str</kwtext>"
#~    echo "  </kw>"
#~  done < ${kwlist_file}
#~  echo '</kwlist>'
#~  ) > $kwsdatadir/kwlist.xml || exit 1
#~
#-(
#-echo '<kwlist ecf_filename="kwlist.xml" language="" encoding="UTF-8" compareNormalize="lowercase" version="" >'
#-id=1
#-while read line; do
#-  id_str=$( printf "KWS$langid-%04d\n" $id )
#-  echo "  <kw kwid=\"$id_str\">"
#-  echo "    <kwtext>$line</kwtext>"
#-  echo "  </kw>"
#-  id=$(( $id + 1 ))
#-done < ${kwlist_file}
#-echo '</kwlist>'
#-) > $kwsdatadir/kwlist.xml || exit 1
#-
