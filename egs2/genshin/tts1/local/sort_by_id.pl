#!/usr/bin/env perl

use strict;
use warnings;

my @lines = <>;

my @sorted = sort {
    my @a_parts = split(/\s+/, $a);
    my @b_parts = split(/\s+/, $b);
    
    # Compare the second part lexicographically, ignoring case
    my $cmp = lc($a_parts[1]) cmp lc($b_parts[1]);
    return $cmp if $cmp != 0;
    
    # If the second part is the same, compare the first part lexicographically, ignoring case
    return lc($a_parts[0]) cmp lc($b_parts[0]);
} @lines;

print @sorted;