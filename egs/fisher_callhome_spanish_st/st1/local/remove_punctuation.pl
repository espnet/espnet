#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # normalize punctuation
  s/&/ & /g;
  s/%/ %/g;    # for En
  s/\$/\$ /g;  # for En
  s/`/'/g;     # for En
  s/´/'/g;     # for En

  # reserve some punctuations
  s/'/apostrophe/g;
  s/&/ampersand/g;
  s/\$/dollarsign/g;  # for En
  s/%/percentsign/g;  # for En

  # remove punctuation
  s/[[:punct:]]/ /g;

  # reserve some punctuations
  s/apostrophe/'/g;
  s/ampersand/&/g;
  s/dollarsign/\$/g;  # for En
  s/percentsign/%/g;  # for En

  # remove noisy parts
  s/background speech//g;
  s/background noise//g;
  s/noise//g;
  s/laughter//g;
  s/lang english online//g;
  s/foreign//g;

  # remove whitespace
  s/\s+/ /g;
  s/^\s+//;
  s/\s+$//;

  print "$_\n";
}
