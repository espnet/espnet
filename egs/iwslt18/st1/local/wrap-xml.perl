#!/usr/bin/perl -w

use strict;
no warnings;

my ($language,$src,$system) = @ARGV;
die("wrapping frame not found ($src)") unless -e $src;
$system = "Edinburgh" unless $system;

open(SRC,$src);
my @OUT = <STDIN>;
chomp(@OUT);
#my @OUT = `cat $decoder_output`;
while(<SRC>) {
    chomp;
    if (/^<srcset/) {
        # s/<srcset/<tstset trglang="$language" sysid="$system"/;
        s/<srcset/<srcset trglang="$language" sysid="$system"/;
    }
    elsif (/^<\/srcset/) {
        # s/<\/srcset/<\/tstset/;
        s/<\/srcset/<\/srcset/;
    }
    elsif (/^<doc/i) {
        s/<doc/<doc sysid="$system"/i;
    }
    elsif (/<seg/) {
        my $line = shift(@OUT);
        $line = "" if $line =~ /NO BEST TRANSLATION/;
        if (/<\/seg>/) {
            s/(<seg[^>]+> *).*(<\/seg>)/$1$line  $2/;
        }
        else {
            s/(<seg[^>]+> *)[^<]*/$1$line/;
        }
    }
    print $_."\n";
}
