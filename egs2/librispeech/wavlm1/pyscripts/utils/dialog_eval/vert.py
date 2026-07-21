# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import warnings
from collections import Counter
from fractions import Fraction

import nltk
import numpy as np
from nltk.translate.bleu_score import (
    SmoothingFunction,
    brevity_penalty,
    closest_ref_length,
    modified_precision,
)


def corpus_bleu(
    list_of_references,
    hypotheses,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
    averaging_mode="geometric",
    no_length_penalty=False,
):
    """
    Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all
    the hypotheses and their respective references.

    Instead of averaging the sentence level BLEU scores (i.e. marco-average
    precision), the original BLEU metric (Papineni et al. 2002) accounts for
    the micro-average precision (i.e. summing the numerators and denominators
    for each hypothesis-reference(s) pairs before the division).

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...          'because', 'he', 'read', 'the', 'book']

    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> corpus_bleu(list_of_references, hypotheses) # doctest: +ELLIPSIS
    0.5920...

    The example below show that corpus_bleu() is different from averaging
    sentence_bleu() for hypotheses

    >>> score1 = sentence_bleu([ref1a, ref1b, ref1c], hyp1)
    >>> score2 = sentence_bleu([ref2a], hyp2)
    >>> (score1 + score2) / 2 # doctest: +ELLIPSIS
    0.6223...

    :param list_of_references: a corpus of lists of reference
    sentences, w.r.t. hypotheses
    :type list_of_references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :param smoothing_function:
    :type smoothing_function: SmoothingFunction
    :param auto_reweigh: Option to re-normalize the weights uniformly.
    :type auto_reweigh: bool
    :return: The corpus-level BLEU score.
    :rtype: float
    """
    # Before proceeding to compute BLEU, perform sanity checks.

    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), (
        "The number of hypotheses and their reference(s) should be the " "same "
    )

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    if no_length_penalty and averaging_mode == "geometric":
        bp = 1.0
    elif no_length_penalty and averaging_mode == "arithmetic":
        bp = 0.0
    else:
        assert not no_length_penalty
        assert (
            averaging_mode != "arithmetic"
        ), "Not sure how to apply length penalty when aurithmetic mode"
        bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [
        Fraction(p_numerators[i], p_denominators[i], _normalize=False)
        for i, _ in enumerate(weights, start=1)
    ]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(
        p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
    )

    if averaging_mode == "geometric":
        s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
        s = bp * math.exp(math.fsum(s))
    elif averaging_mode == "arithmetic":
        s = (w_i * p_i for w_i, p_i in zip(weights, p_n))
        s = math.fsum(s)

    return s


def sentence_bleu(
    references,
    hypothesis,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
    averaging_mode="geometric",
    no_length_penalty=False,
):
    return corpus_bleu(
        [references],
        [hypothesis],
        weights,
        smoothing_function,
        auto_reweigh,
        averaging_mode,
        no_length_penalty,
    )


def get_target_sequences(manifest, ground_truth, to_take=1000):
    import json
    import pathlib

    with open(ground_truth, "r") as fin:
        original_continuations = json.loads(fin.read())

    sequence2length = [(k, v[0]) for k, v in original_continuations.items()]
    assert all(float(v) >= 6.0 for (_, v) in sequence2length)  # 6 seconds

    sequence2length.sort(key=lambda x: x[1])
    to_take_sequences = set(v[0] for v in sequence2length[:to_take])
    to_take_ids = []

    with open(manifest, "r") as f:
        f.readline()

        for i, line in enumerate(f.readlines()):
            seq_id = line.split()[0]
            seq_id = pathlib.Path(seq_id).name.split("__")[0]

            if seq_id in to_take_sequences:
                to_take_ids.append(i)

    print(f"Took {len(to_take_ids)} ids")
    return set(to_take_ids)


def get_self_bleu(utterances, averaging_mode, weights):
    self_bleu = []

    for i in range(len(utterances)):
        hypo = utterances[i]
        rest = utterances[:i] + utterances[i + 1 :]

        self_bleu.append(
            sentence_bleu(
                rest,
                hypo,
                weights,
                no_length_penalty=True,
                averaging_mode=averaging_mode,
            )
        )

    return self_bleu


def get_self_bleu2_arithmetic(utterances):
    weights = (0.5, 0.5)  # equal weight for unigrams and bigrams
    return get_self_bleu(utterances, averaging_mode="arithmetic", weights=weights)


def get_self_bleu2_geometric(utterances):
    weights = (0.5, 0.5)
    return get_self_bleu(utterances, averaging_mode="geometric", weights=weights)


def get_auto_bleu2_arithmetic(utterances):
    weights = (0.5, 0.5)
    return [auto_bleu(u, mean_mode="arithmetic", weights=weights) for u in utterances]


def get_auto_bleu2_geometric(utterances):
    weights = (0.5, 0.5)
    return [auto_bleu(u, mean_mode="geometric", weights=weights) for u in utterances]


def get_auto_bleu3_geometric(utterances):
    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return [auto_bleu(u, mean_mode="geometric", weights=weights) for u in utterances]


def get_auto_bleu3_arithmetic(utterances):
    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return [auto_bleu(u, mean_mode="arithmetic", weights=weights) for u in utterances]


def get_self_bleu3_arithmetic(utterances):
    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return get_self_bleu(utterances, averaging_mode="arithmetic", weights=weights)


def get_self_bleu3_geometric(utterances):
    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return get_self_bleu(utterances, averaging_mode="geometric", weights=weights)


def auto_bleu(sentence, weights, mean_mode="arithmetic"):
    if len(sentence) <= 1:
        return 0

    N = len(weights)

    bleu_n = np.zeros([N])
    for n in range(N):
        targ_ngrams = list(nltk.ngrams(sentence, n + 1))
        for p in range(len(targ_ngrams)):
            left = sentence[:p]
            right = sentence[(p + n + 1) :]
            rest_ngrams = list(nltk.ngrams(left, n + 1)) + list(
                nltk.ngrams(right, n + 1)
            )
            # compute the nb of matching ngrams
            bleu_n[n] += targ_ngrams[p] in rest_ngrams
        bleu_n[n] /= len(targ_ngrams)  # average them to get a proportion

    weights = np.array(weights)
    if mean_mode == "arithmetic":
        return (bleu_n * weights).sum()
    elif mean_mode == "geometric":
        return (bleu_n**weights).prod()
    else:
        raise ValueError(f"Unknown agggregation mode {mean_mode}")


def run_f(task_params):
    f, terms = task_params
    return f(terms)
