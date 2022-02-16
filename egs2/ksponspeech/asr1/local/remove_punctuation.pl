#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # remove punctuation except apostrophe
  s/<space>/spacemark/g;  # for scoring
  s/<unk>/unknown1/g;
  s/\[unk\]/unknown2/g;
  s/\+/repeatsym/g;
  s/\//fillersym/g;
  s/'/apostrophe/g;
  s/[[:punct:]]//g;
  s/apostrophe/'/g;
  s/spacemark/<space>/g;  # for scoring
  s/fillersym/\//g;
  s/repeatsym/\+/g;
  s/unknown2/\[unk\]/g;
  s/unknown1/<unk>/g;

  # remove consecutive commas and spaces
  s/\s+/ /g;

  # remove whitespace again
  s/\s+/ /g;
  s/^\s+//;
  s/\s+$//;

  print "$_\n";
}
