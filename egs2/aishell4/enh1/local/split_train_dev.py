#!/usr/bin/env python

# Copyright 2022  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple


def int_or_float_or_numstr(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        assert 0 < value < 1, value
        return Fraction(value)
    elif isinstance(value, (str, Fraction)):
        num = Fraction(value)
        if num.denominator == 1:
            return num.numerator  # int
        else:
            return num
    else:
        raise TypeError("Unsupported value type: %s" % type(value))


def split_train_dev(all_data, num_dev, outfile):
    """Group all paths listed in `datalist` according to their name prefix,
    and split all groups into train and dev subsets.

    Each subset has distinct groups.

    This is designed for splitting same-size groups into train and dev subsets.
    That is, all groups must have exactly the same number of paths.

    Args:
        all_data (Dict[group_name: List[str]]): all paths grouped by their category
        num_dev (int or Fraction): number/percentage of the samples for the dev set
        outfile (str): template path to the ourput file
    """  # noqa: H405

    # all groups must have the same number of paths
    group_id0 = next(iter(all_data.keys()))
    group_length = len(all_data[group_id0])
    for k, v in all_data.items():
        assert len(v) == group_length, (k, len(v), group_length)
    total_num = group_length * len(all_data)

    # determine number of groups for dev subset
    if isinstance(num_dev, int):
        assert num_dev % group_length == 0, (num_dev, group_length)
        num_dev_groups = num_dev // group_length
    elif isinstance(num_dev, Fraction):
        num_dev_groups = num_dev * total_num / group_length
        if num_dev_groups.denominator == 1:
            num_dev_groups = num_dev_groups.numerator
        else:
            num_dev_groups = round(num_dev_groups)
            print(
                "Warning: num_dev_groups is rounded to the nearest integer "
                f"{num_dev_groups}."
            )
    else:
        raise TypeError("Unsupported data type: %s" % type(num_dev))

    groups = list(all_data.keys())
    random.shuffle(groups)
    dev_groups = groups[:num_dev_groups]
    train_groups = groups[num_dev_groups:]

    outdir = Path(outfile).expanduser().resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    assert "{}" in outfile, outfile
    with Path(outfile.format("dev")).open("w") as out:
        for room in dev_groups:
            for wav in all_data[room]:
                out.write(wav)
    with Path(outfile.format("train")).open("w") as out:
        for room in train_groups:
            for wav in all_data[room]:
                out.write(wav)


def split_train_dev_v2(
    all_data, num_dev, outfile, allowed_deviation=0, max_solutions=50
):
    """Group all paths listed in args.datalist according to their name prefix,
    and split all groups into train and dev subsets.

    Each subset has distinct groups.

    This is designed for splitting similar-size groups into train and dev subsets.
    That is, all groups must have, if not the same, a similar number of paths.

    Args:
        all_data (Dict[group_name: List[str]]): all paths grouped by their category
        num_dev (int or Fraction): number/percentage of the samples for the dev set
        outfile (str): template path to the ourput file
        allowed_deviation (int): Allowed number of samples for the final dev split to be
                                less than or more than the specified `num_dev`
        max_solutions (int): maximum number of possible coin change solutions to search
    """  # noqa: H405

    lengths = [len(v) for v in all_data.values()]
    total_length = sum(lengths)
    mean_length = total_length / len(lengths)
    print(
        f"len(group_lengths)={len(lengths)}\n"
        f"max(group_lengths)={max(lengths)}, min(group_lengths)={min(lengths)}, "
        f"mean(group_lengths)={mean_length:.2f}\n"
    )

    # determine number of groups for dev subset
    if isinstance(num_dev, int):
        num_dev_samples = num_dev
    elif isinstance(num_dev, Fraction):
        num_dev_samples = num_dev * total_length
        if num_dev_samples.denominator == 1:
            num_dev_samples = num_dev_samples.numerator
        else:
            num_dev_samples = round(num_dev_samples)
            print(
                "Warning: num_dev_samples is rounded to the nearest integer "
                f"({num_dev_samples})."
            )
    else:
        raise TypeError("Unsupported data type: %s" % type(num_dev))

    # Solve this assignment problem like the recursive Coin Change Problem
    choices = find_all_coin_change_ways(
        lengths,
        num_dev_samples,
        allowed_deviation=allowed_deviation,
        max_solutions=max_solutions,
    )
    if len(choices) == 0:
        raise ValueError(
            "Current find an exact solution to match num_dev_samples (=%d)\n"
            "Please modify --num_dev or consider using a larger value for "
            "--allowed_deviation" % num_dev_samples
        )

    choice = random.choice(choices)
    groups_count = Counter(choice)
    pool = defaultdict(list)
    groups = list(all_data.keys())
    for i, length in enumerate(lengths):
        if length in groups_count:
            pool[length].append(i)

    selected_idx = []
    dev_groups = []
    for length, num in groups_count.items():
        name_idxs = random.sample(pool[length], num)
        selected_idx.extend(name_idxs)
        for idx in name_idxs:
            dev_groups.append(groups[idx])
    for idx in sorted(selected_idx, reverse=True):
        groups.pop(idx)
    train_groups = groups

    outdir = Path(outfile).expanduser().resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    assert "{}" in outfile, outfile
    with Path(outfile.format("dev")).open("w") as out:
        for room in dev_groups:
            for wav in all_data[room]:
                out.write(wav)
    with Path(outfile.format("train")).open("w") as out:
        for room in train_groups:
            for wav in all_data[room]:
                out.write(wav)


def find_all_coin_change_ways(
    coins: List, amount: int, allowed_deviation: int = 0, max_solutions=50
):
    def coin_change(unique_coins: List, amount: int, tmp_ret: Tuple = ()):
        """Search in ascending order."""
        if len(unique_coins) == 0 or len(all_combinations) > max_solutions:
            return
        biggest_coin, rest_coins = unique_coins[0], unique_coins[1:]
        if allowed_deviation > 0:
            num = math.ceil(amount / biggest_coin)
        else:
            num = amount // biggest_coin
        for i in range(1 + num):
            if i > coin_num[biggest_coin]:
                break
            remainder = amount - biggest_coin * i
            if abs(remainder) <= allowed_deviation:
                new_combo = tmp_ret + (biggest_coin,) * i
                if allowed_deviation > 0:
                    if len(all_combinations) == 0 or new_combo != all_combinations[-1]:
                        all_combinations.append(new_combo)
                    coin_change(rest_coins, remainder, tmp_ret + (biggest_coin,) * i)
                else:
                    all_combinations.append(new_combo)
                    break
            elif remainder > 0:
                coin_change(rest_coins, remainder, tmp_ret + (biggest_coin,) * i)
            else:
                break

    def coin_change_v2(unique_coins: List, amount: int, tmp_ret: Tuple = ()):
        """Search in descending order."""
        if len(unique_coins) == 0 or len(all_combinations) > max_solutions:
            return
        biggest_coin, rest_coins = unique_coins[0], unique_coins[1:]
        if allowed_deviation > 0:
            num = min(coin_num[biggest_coin], math.ceil(amount / biggest_coin))
        else:
            num = min(coin_num[biggest_coin], amount // biggest_coin)
        for i in range(num, -1, -1):
            remainder = amount - biggest_coin * i
            if abs(remainder) <= allowed_deviation:
                new_combo = tmp_ret + (biggest_coin,) * i
                if allowed_deviation > 0:
                    if len(all_combinations) == 0 or new_combo != all_combinations[-1]:
                        all_combinations.append(new_combo)
                    length = len(all_combinations)
                    coin_change_v2(rest_coins, remainder, tmp_ret + (biggest_coin,) * i)
                    if len(all_combinations) == length:
                        break
                else:
                    all_combinations.append(new_combo)
            elif remainder > 0:
                coin_change_v2(rest_coins, remainder, tmp_ret + (biggest_coin,) * i)

    coin_num = Counter(coins)
    unique_coins = sorted(coin_num.keys(), reverse=True)  # in descending order
    all_combinations = []
    if unique_coins[0] * coin_num[unique_coins[0]] < amount:
        coin_change_v2(unique_coins, amount)
    else:
        coin_change(unique_coins, amount)
    return all_combinations
