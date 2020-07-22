#!/usr/bin/env perl
#
# Google Drive direct download of big files
# ./gdown.pl 'gdrive file url' ['desired file name']
#
# v1.0 by circulosmeos 04-2014.
# v1.1 by circulosmeos 01-2017.
# v1.2, v1.3, v1.4 by circulosmeos 01-2019, 02-2019.
# //circulosmeos.wordpress.com/2014/04/12/google-drive-direct-download-of-big-files
# Distributed under GPL 3 (//www.gnu.org/licenses/gpl-3.0.html)
#
use strict;
use POSIX;

my $TEMP='gdown.cookie.temp';
my $COMMAND;
my $confirm;
my $check;
sub execute_command();

my $URL=shift;
die "\n./gdown.pl 'gdrive file url' [desired file name]\n\n" if $URL eq '';

my $FILENAME=shift;
$FILENAME='gdown.'.strftime("%Y%m%d%H%M%S", localtime).'.'.substr(rand,2) if $FILENAME eq '';

if ($URL=~m#^https?://drive.google.com/file/d/([^/]+)#) {
    $URL="https://docs.google.com/uc?id=$1&export=download";
}
elsif ($URL=~m#^https?://drive.google.com/open\?id=([^/]+)#) {
    $URL="https://docs.google.com/uc?id=$1&export=download";
}

execute_command();

while (-s $FILENAME < 100000) { # only if the file isn't the download yet
    open fFILENAME, '<', $FILENAME;
    $check=0;
    foreach (<fFILENAME>) {
        if (/href="(\/uc\?export=download[^"]+)/) {
            $URL='https://docs.google.com'.$1;
            $URL=~s/&amp;/&/g;
            $confirm='';
            $check=1;
            last;
        }
        if (/confirm=([^;&]+)/) {
            $confirm=$1;
            $check=1;
            last;
        }
        if (/"downloadUrl":"([^"]+)/) {
            $URL=$1;
            $URL=~s/\\u003d/=/g;
            $URL=~s/\\u0026/&/g;
            $confirm='';
            $check=1;
            last;
        }
    }
    close fFILENAME;
    die "Couldn't download the file :-(\n" if ($check==0);
    $URL=~s/confirm=([^;&]+)/confirm=$confirm/ if $confirm ne '';

    execute_command();
}

unlink $TEMP;

sub execute_command() {
    $COMMAND="wget --progress=dot:giga --no-check-certificate --load-cookie $TEMP --save-cookie $TEMP \"$URL\"";
    $COMMAND.=" -O \"$FILENAME\"" if $FILENAME ne '';
    system ( $COMMAND );
    return 1;
}
