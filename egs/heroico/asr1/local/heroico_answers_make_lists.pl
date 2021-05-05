#!/usr/bin/env perl

# Copyright 2020 ARL (John Morgan)
# Apache 2.0.

# heroico_answers_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

# set variables
my $tmpdir = "data/local/tmp/heroico";
my $wav_list = "$tmpdir/wav_list.txt";
my $wav_scp = "$tmpdir/answers/wav.scp";
my $utt2spk = "$tmpdir/answers/utt2spk";
my $txt = "$tmpdir/answers/text";
my %prompts = ();

system "mkdir -vp $tmpdir/answers";

# store prompts in hash
LINEA: while ( my $line = <> ) {
    chomp $line;
    my ($num,$sent) = split /\t/, $line, 2;
    next LINEA if $sent =~ /^$/;
    my ($volume,$directories,$file) = File::Spec->splitpath( $num );
    my @dirs = split /\//, $directories;
    # get the speaker number
    my $s = $dirs[-1];
    # pad the speaker number with zeroes
    my $spk = "";
    if ( $s < 10 ) {
        $spk = '000' . $s;
    } elsif ( $s < 100 ) {
        $spk = '00' . $s;
    } elsif ( $s < 1000 ) {
        $spk = '0' . $s;
    }
    # pad the filename with zeroes
    my $fn = "";
    if ( $file < 10 ) {
        $fn = '000' . $file;
    } elsif ( $file < 100 ) {
        $fn = '00' . $file;
    } elsif ( $file < 1000 ) {
        $fn = '0' . $file;
    }
    # the utterance name
    my $utt = $spk . '_' . $fn;
    $prompts{$utt} = $sent;
}

open my $WAVLIST, '<', $wav_list or croak "problem with $wav_list $!";
open my $WAVSCP, '+>', $wav_scp or croak "problem with $wav_scp $!";
open my $UTTSPK, '+>', $utt2spk or croak "problem with $utt2spk $!";
open my $TXT, '+>', $txt or croak "problem with $txt $!";

LINE: while ( my $line = <$WAVLIST> ) {
    chomp $line;
    next LINE unless ( $line =~ /Answers/ );
    next LINE if ( $line =~ /Recordings/ );
    my ($volume,$directories,$file) = File::Spec->splitpath( $line );
    my @dirs = split /\//, $directories;
    my $r = basename $line, ".wav";
    my $s = $dirs[-1];
    my $spk = "";
    # pad with zeroes
    if ( $s < 10 ) {
        $spk = '000' . $s;
    } elsif ( $s < 100 ) {
        $spk = '00' . $s;
    } elsif ( $s < 1000 ) {
        $spk = '0' . $s;
    }
    # pad the file name with zeroes
    my $rec = "";
    if ( $r < 10 ) {
        $rec = '000' . $r;
    } elsif ( $r < 100 ) {
        $rec = '00' . $r;
    } elsif ( $r < 1000 ) {
        $rec = '0' . $r;
    }
    my $rec_id = $spk . '_' . $rec;
    if ( exists $prompts{$rec_id} ) {
        print $TXT "$rec_id $prompts{$rec_id}\n";
    } elsif ( defined $rec_id ) {
        warn  "warning: problem\t$rec_id";
        next LINE;
    } else {
        croak "$line";
    }

    print $WAVSCP "$rec_id sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
    print $UTTSPK "$rec_id $spk\n";
}
close $TXT;
close $WAVSCP;
close $UTTSPK;
close $WAVLIST;
