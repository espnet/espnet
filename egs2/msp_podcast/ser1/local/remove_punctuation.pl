#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  # remove punctuation except apostrophe
  # extra spaces will be removed later in "normalize_punctuation.pl"
  s/([\.\,\?\!\-\:\;\"])/ $1 /g;
  s/[\.\,\?\!\-\:\;\"]//g;
  print "$_";
}
