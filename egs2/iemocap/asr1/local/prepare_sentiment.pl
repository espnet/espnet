#!/usr/bin/env perl
#
# Yushi Ueda, Carnegie Mellon University (2022)

use warnings;
use strict;

while (@ARGV) {
    $_ = shift;
    /^-b$/ && ($| = 1, next); # not buffered (flush each line)
}

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

while(<STDIN>) {
  # remove punctuation except apostrophes
  s/([\.\,\?\!\-\:\;\"])/ $1 /g;
  s/[\.\,\?\!\-\:\;\"]//g;
  # remove tag (e.g. [LAUGHTER])
  s/\[.+\]//g;
  # Detect valid apostrophe cases and split those into a two words
  s/([A-Za-z])\'([A-Za-z])/$1 \'$2/g;
  # Clean up special cases of standalone apostrophes
  s/([A-Za-z])\' /$1 /g;
  # remove extra spaces
  s/ +/ /g;
  # lowercasing
  print lc($_);
}
