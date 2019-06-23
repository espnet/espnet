#!/usr/bin/perl

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# for ST-TED
# NOTE: this script is applied only for the training set

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # normalize punctuation
  s/\// \/ /g;  # insert space at both ends of /
  s/，/,/g; # ， -> ,
  s/\, \,/\,/g;  # , , -> ,
  s/\.\. \./\.\.\./g;  # .. . -> ...
  s/\.\.\.+/\.\.\./g;  # ..... -> ...
  s/- -/--/g;  # - - -> --
  s/--+/--/g;  # ----- -> --
  s/\&quot/\"/g;  # incomplete xml

  # remove noisy parts
 s/\<br\>/ /g;
 s/\<br\s*\/\s*\>/ /g;

  # remove punctuation
  s/_/ /g;
  s/\[/ /g;
  s/\]/ /g;
  s/\\/ /g;
  # s/\;/ /g;
  s/\*/ /g;
  s/\^/ /g;

  # remove noisy punctuation in the first character
  my $count;
  for ($count = 0; $count < 4; $count++){
      s/^\s+//g;
      s/^\,\s*//g;
      s/^\.+\s*//g;
      ###
      s/^\?\s*//g;
      s/^\"\s*//g;
      s/^\'\s*//g;
      s/^-+\s*//g;
  }

  # remove consecutive whitespaces
  s/\s+/ /g;

  # remove the first and last whitespaces
  s/^\s+//g;
  s/\s+$//g;

  print "$_\n";
}
