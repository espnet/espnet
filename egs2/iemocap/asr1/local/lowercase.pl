#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.
# Modified for espnet2 iemocap recipe
# Yushi Ueda, Carnegie Mellon University (2021)

use warnings;
use strict;

while (@ARGV) {
    $_ = shift;
    /^-b$/ && ($| = 1, next); # not buffered (flush each line)
}

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

while(<STDIN>) {
  if ($_ =~ /(^Ses.+[0-9]{3})\s(<.+$)/){
    my $word = lc($2);
    print "$1 $word\n";
  }
}
