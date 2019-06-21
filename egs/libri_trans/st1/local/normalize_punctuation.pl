#!/usr/bin/perl

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# for Librispeech-French
# NOTE: this script is applied only for the training set

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # remove citation
  s/\[[0-9*\s]+\]/ /g;

  # remove A.D.
  s/\([0-9\s]+\)/ /g;

  # insert whitespace on both sides of brachets
  s/\(/ \(/g;
  s/\)/\) /g;
  s/\[/ \[/g;
  s/\]/\] /g;

  # normalize punctuation
  s/\, \,/\,/g;  # , , -> ,
  s/\.\. \./\.\.\./g;  # .. . -> ...
  s/\.\.\.+/\.\.\./g;  # ..... -> ...
  s/- -/--/g;  # - - -> --
  s/--+/--/g;  # ----- -> --
  s/(\" )+/\" /g;  # (" ) ->
  s/\"\"/\"/g;  # "" => "

  # remove noisy parts
  s/file .+ FLAVIUS JOSEPHE/ /g;  # 83-138254-0097 in train.fr
  s/700000800000/ /g;  # 1184-121085-0035 in train.en
  s/700 000 000 000/ /g;  # 1184-121085-0035 in train.fr.gtranslate
  s/\&gt//;  # incomplete xml
  s/\&lt//;  # incomplete xml

  # remove noisy punctuation
  s/_/ /g;
  s/\(/ /g;
  s/\)/ /g;
  s/\</ /g;
  s/\>/ /g;
  s/\[/ /g;
  s/\]/ /g;
  s/\{/ /g;
  s/\}/ /g;
  s/\\/ /g;
  s/\// /g;
  # s/\;/ /g; # used for evaluation
  s/\*/ /g;
  s/\^/ /g;

  # remove noisy punctuation in the first character
  my $count;
  for ($count = 0; $count < 4; $count++){
      s/^\s+//g;
      s/^\,\s*//g;
      s/^\.+\s*//g;
      s/^!\s*//g;
      ###
      s/^\?\s*//g;
      s/^\"\s*//g;
      s/^\'\s*//g;
      s/^-+\s*//g;
      s/^:\s*//g;
      s/^&\s*//g;
      s/^\%\s*//g;
  }
  s/^\.\.!\s//;
  s/^\.\.\?\s//;

  # remove consecutive whitespaces
  s/\s+/ /g;

  # remove the first and last whitespaces
  s/^\s+//g;
  s/\s+$//g;

  print "$_\n";
}
