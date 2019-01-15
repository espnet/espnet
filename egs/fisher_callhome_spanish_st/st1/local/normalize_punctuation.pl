#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # remove brachets and inside
  s/\([^\)]+\)/ /g;
  s/\[[^\]]+\]/ /g;

  # normalize punctuation
  s/_/ /g;
  s/&/ & /g;
  s/--/ - /g;
  s/%/ %/g;    # for En
  s/\$/\$ /g;  # for En
  s/`/'/g;     # for En
  s/´/'/g;     # for En

  # remove noisy parts
  s/noise//g;
  s/laughter//g;
  s/background noise//g;
  s/background speech//g;

  # fisher_train
  s/i\/he/i/g;
  s/i\/she/i/g;
  s/ \/\?/\?/g;
  s/ \/ / /g;
  s/a\/c//g;
  s/stay\//stay/g;
  s/boys\//boys/g;
  s/right\//right/g;
  s/follow\//follow/g;
  s/Jose\/Josefina/Jose/g;
  s/welfare\/foreign/welfare/g;
  s/\<foreign lang=\"English\"//g;
  s/\/foreign//g;
  s/\<plural\>//g;
  s/\<barely makes any sense\>//g;
  s/\<kind of a weird phrase\>//g;
  s/\<last word does not fit there\>//g;
  s/\<players with the meaning of singers\>//g;
  s/\<this phrase barely made any sense whatsoever\>//g;
  s/\<colorcito does not exist as a word so I have no ideea what he means about that\>//g;
  s/\<foreign//g;
  s/foreign\>//g;

  # fisher_dev
  s/her\/his/her/g;
  s/o\//o/g;
  s/co\//co/g;
  s/L \/ //g;
  s/\<\?\?\?\>//g;
  s/\<from Texas\>//g;
  s/\<weird phrase\>//g;
  s/\<this makes no sense\>//g;
  s/Salvador\>/Salvador/g;

  # fisher_dev2
  s/A\/C//g;
  s/She\/he/She/g;
  s/you\/he/you/g;
  s/you\/she/you/g;
  s/Um\//Um/g;
  s/name\//name/g;
  s/American\//American/g;
  s/\<\?\>//g;
  s/\<metaphoric meaning\>//g;
  s/\<missing text \? \>//g;
  s/\<broken phrase but I tried to guess what would it mean if it was complete\>//g;

  # fisher_test
  s/she\/he/she/g;
  s/her\/him/her/g;
  s/is\//is/g;
  s/and\/or/and/g;
  s/Then\/Well/Then/g;
  s/fine\/well/fine/g;
  s/Likewise\/Equally/Likewise/g;
  s/boyfriend\/girlfriend/boyfriend/g;
  s/living room \/ dining room/living room/g;
  s/\<very bad phrase\>//g;
  s/\<poorly written phrase\>//g;
  s/\<this phrase barely even made sense\>//g;
  s/\<very poorly written phrase but I think this is what was supposed to mean\>//g;
  s/what\)\)/what/g;

  # callhome_train
  s/-*-//g;

  # callhome_devtest
  s/He\/she/He/g;
  s/he\/she/he/g;
  s/he\/se/he/g;
  s/his\/her/his/g;
  s/him\/her/him/g;
  s/right'$/right?/;

  # callhome_evltest
  s/so\//so/g;

  # remove punctuation
  s/\(/ /g;
  s/\)/ /g;
  s/\</ /g;
  s/\>/ /g;
  s/\[/ /g;
  s/\]/ /g;
  s/\\/ /g;
  s/\// /g;
  s/:/ /g;
  s/\;/ /g;
  s/~/ /g;
  s/=/ /g;
  s/\}/ /g;
  s/\·/ /g;
  s/\¨/ /g;
  s/\*/ /g; # for callhome_train

  # remove consecutive commas and spaces
  s/\.+/ /g;
  s/\s+/ /g;

  # remove last bar (except for partial words)
  s/\s+$//;
  s/\s-$//;

  # remove whitespace again
  s/\s+/ /g;
  s/^\s+//;
  s/\s+$//;

  print "$_\n";
}
