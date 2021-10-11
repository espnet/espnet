#!/usr/bin/env perl
use strict;
use warnings;
use File::Basename;
use utf8;

my $input_csv = $ARGV[0] or die "Need to get CSV file on the command line\n";
my $ses_id = basename($input_csv, '.csv');
my $wav_file = $ARGV[1] or die "Need to get wav file on the command line\n";
my $output_wav_scp = "data/tmp/wav.scp";
my $output_utt2spk = "data/tmp/utt2spk";
my $output_text = "data/tmp/text";
my $output_segments = "data/tmp/segments";

open(my $csv, '<', $input_csv) or die "Couldn't open '$input_csv' $!\n";
open (my $wav_scp, '>>', $output_wav_scp) or die "Couldn't open file '$output_wav_scp' $!\n";
open (my $utt2spk, '>>', $output_utt2spk) or die "Couldn't open file '$output_utt2spk' $!\n";
open (my $text, '>>', $output_text) or die "Couldn't open file '$output_text' $!\n";
open (my $segments, '>>', $output_segments) or die "Couldn't open file '$output_segments' $!\n";

# wav.scp
print $wav_scp "${ses_id} sox ${wav_file} -t wav -c 1 - rate 16000 |\n";

my @lines = <$csv>;
my $i;
# preprocess words
for ($i=@lines-1; $i > 0; $i--){
  chomp($lines[$i]);
  my @fields = split "," , $lines[$i];
  if ($fields[0] eq "*"){
    my @words = @fields[5 .. $#fields];
    chomp($lines[$i-1]);
    $lines[$i-1] = $lines[$i-1] . "," . join(',',@words);
  }
}

# create files
for ($i=0; $i<@lines; $i++){
  chomp($lines[$i]);
  my @fields = split "," , $lines[$i];
  my @tag = split "/" , $fields[2];
  if($#fields >= 5 && $fields[0] ne "*" && $tag[0] =~ /[A-Z]{2}-[A-Z]{2,3}/ && $fields[3] =~ /^[0-9]+$/ && $fields[4] =~ /^[0-9]+$/ && $fields[4] > $fields[3] && $fields[4] - $fields[3] < 400000000){
    my $start_time = $fields[3] / 10000000;
    my $end_time = $fields[4] / 10000000;
    my @words = @fields[5 .. $#fields];
    my $joined_words = join(' ',@words);
    # clean words
    $joined_words =~ s/ +/ /g;
    $joined_words =~ s/\?/ /g;
    $joined_words =~ s/\([LFR]([^a-zA-Z]+)\)/$1/g;
    $joined_words =~ s/\(P\)//g;
    $joined_words =~ s/\[.+\]//g;
    $joined_words =~ s/ +/ /g;
    $joined_words =~ s/^ //g;
    $joined_words =~ s/ $//g;
    # utt2spk
    print $utt2spk "${ses_id}_$fields[3]_$fields[4] ${ses_id}_$fields[3]_$fields[4]\n";
    # segments
    print $segments "${ses_id}_$fields[3]_$fields[4] ${ses_id} ${start_time} ${end_time}\n";
    # text
    print $text "${ses_id}_$fields[3]_$fields[4] $tag[0] $joined_words\n";
  }
}

close $csv or die "Couldn't close '$input_csv' $!\n";
close $wav_scp or die "Couldn't close file '$output_wav_scp' $!\n";
close $utt2spk or die "Couldn't close file '$output_utt2spk' $!\n";
close $text or die "Couldn't close file '$output_text' $!\n";
close $segments or die "Couldn't close file '$output_segments' $!\n";
