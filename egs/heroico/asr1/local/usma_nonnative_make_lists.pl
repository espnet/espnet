#!/usr/bin/env perl

# Copyright 2020 ARL (John Morgan)
# Apache 2.0.

#usma_nonnative_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

# Set variables
my $tmpdir = "data/local/tmp/usma/nonnative";
my $wav_list = "$tmpdir/wav_list.txt";
my $wav_scp = "$tmpdir/wav.scp";
my $utt_to_spk = "$tmpdir/utt2spk";
my $txt = "$tmpdir/text";
my %prompts = ();

# store prompts in hash
LINEA: while ( my $line = <> ) {
    chomp $line;
    my ($num,$sent) = split /\t/, $line, 2;
    $prompts{$num} = $sent;
}

open my $WAV, '<', $wav_list or croak "problem with $wav_list $!";
open my $WAVSCP, '+>', $wav_scp or croak "problem with $wav_scp $!";
open my $UTTSPK, '+>', $utt_to_spk or croak "problem with $utt_to_spk $!";
open my $TXT, '+>', $txt or croak "problem with $txt $!";

LINE: while ( my $line = <$WAV> ) {
    chomp $line;
    next LINE unless ( $line =~ /nonnative/ );
    my ($volume,$directories,$file) = File::Spec->splitpath( $line );
    my @dirs = split /\//, $directories;
    my $r = basename $line, ".wav";
    next LINE unless ( $r =~ /^s/ );
    my $s = $dirs[-1];
    my ($nativeness,$gender,$country,$weight,$age,$height,$dlpt,$idx) = split /\-/, $s, 9;
    $s = $nativeness . '_' . $gender . '_' . $country . '_' . $weight . '_' . $age . '_' . $height . '_' . $dlpt . '_' . $idx;
    my $rid = $s . '_' . $r;
    if ( exists $prompts{$r} ) {
        print $TXT "$rid $prompts{$r}\n";
    } elsif ( defined $rid ) {
        warn  "problem\t$rid";
        next LINE;
    } else {
        croak "$line";
    }

    print $WAVSCP "$rid sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
    print $UTTSPK "$rid $s\n";
}
close $TXT;
close $WAVSCP;
close $UTTSPK;
close $WAV;
