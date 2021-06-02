ali-to-phones final.mdl ark:"gunzip -c ali.*.gz|" ark,t:text.int
../../utils/int2sym.pl -f 2- phones.txt text.int > text
