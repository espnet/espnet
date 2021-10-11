#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  # remove tag (e.g. [LAUGHTER])
  s/\[.+\]//g;
  print "$_";
}
