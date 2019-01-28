#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # remove punctuation except apostrophe
  s/'/apostrophe/g;
  s/[[:punct:]]//g;
  s/apostrophe/'/g;

  # remove consecutive commas and spaces
  s/\s+/ /g;

  # remove whitespace again
  s/\s+/ /g;
  s/^\s+//;
  s/\s+$//;

  print "$_\n";
}
