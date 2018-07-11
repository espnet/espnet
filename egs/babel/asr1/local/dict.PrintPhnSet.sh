#!/bin/bash 

first_variant_only=""
mlf=""
for i in $*; do
    case "$1" in
        -first-variant-only)
            first_variant_only="T"
            shift
            ;;
	-mlf)
	    mlf=$2
	    shift
	    shift
            ;;
	-text)
	    text=$2
	    shift
	    shift
            ;;
	-*)
            echo Unknown option $1
            exit
            ;;
        *)
            break
            ;;
    esac
done

in=$*
[ -z "$in" ]     && in=/dev/stdin
[ "$in" == "-" ] && in=/dev/stdin


if [ ! -z "$mlf" ] || [ ! -z "$text" ]; then
### Expand Words in MLF by DICT and print stats 
### Note: this option take first_variant_only="T"
    awk -v mlf="$mlf" -v ttext="$text" '
{
  wrd=$1; 
  if (!(wrd in DCT)) {
   $1=""; DCT[wrd]=$0;
  }


} 
    
END{

if(mlf){
 getline <mlf;getline <mlf; ## Cut header and first file name
 while( getline <mlf) {
   if( /^[.]$/ ) { # Cut end and filename 
    getline <mlf; 
    if (!(getline <mlf)) break; 
   }

   wrd=$1
   if (wrd in DCT){
     nS=split(DCT[wrd],S); # Split into phonemes
     for(i=1;i<=nS;i++) PHN[S[i]]++;
   } else{
    print "Warning: word > " wrd " < is not in dictionary" > "/dev/stderr" 
   }
 }
}
if(ttext){
 while( getline <ttext) {
   for(i=2;i<=NF;i++){
    wrd=$i
    if (wrd in DCT){
      nS=split(DCT[wrd],S); # Split into phonemes
      for(j=1;j<=nS;j++) PHN[S[j]]++;
    } else{
     print "Warning: word > " wrd " < is not in dictionary" > "/dev/stderr" 
    }
   }
 }
}

## Print stats   
for(phn in PHN) print phn "\t " PHN[phn]
}' $in

else  
    awk ${first_variant_only:+-v first_variant_only=1}  '
{ 
  bcount=0;
  if (first_variant_only) {
    wrd=$1
    if (!(wrd in WRD)) { bcount=1; WRD[wrd]=1}
  } else {
    bcount=1
  }

  if (bcount) { 
    for (i=2;i<=NF;i++) {
      if (!($i in PHN)) { PHN[$i]=1;
      }
    }
  }
}

END{
    for(phn in PHN) print phn
}' $in
fi
