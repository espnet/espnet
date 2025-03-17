#!/usr/bin/env python3
"""test of compute_metrics."""

import pytest
from pyscripts.utils.compute_turn_take_metrics import (
    ModelParam,
    ScoreResult,
    compute_turn_decisions,
    compute_turn_likelihoods,
)

labels = [
    "C",  # Continuation
    "NA",  # Silence
    "IN",  # Interruption
    "BC",  # Backchannel
    "T",  # Turn Change
]


def test_F1_simple():
    ref_arr_simple = [
        "moshi_audio_1\
        0.13,0.59,0.02,0.02,0.25\
        0.06,0.04,0.04,0.07,0.78\
        0.52,0.25,0.03,0.04,0.16\
        0.67,0.10,0.01,0.01,0.21\
        0.20,0.62,0.02,0.02,0.14\
        0.84,0.01,0.13,0.02,0.01\
        0.13,0.30,0.04,0.41,0.12\
        "
    ]
    hyp_arr_simple = [
        "moshi_audio_1,0.2,0.24,NA,NA",
        "moshi_audio_1,0.24,0.28,T,NA",
        "moshi_audio_1,0.28,0.32,C,A",
        "moshi_audio_1,0.32,0.36,IN,A",
        "moshi_audio_1,0.36,0.40,T,A",
        "moshi_audio_1,0.40,0.44,C,B",
        "moshi_audio_1,0.44,0.48,BC,B",
    ]
    true_dict = compute_turn_likelihoods(
        ref_arr_simple, ModelParam.min_start_time.value, ModelParam.chunk_length.value
    )
    pred_dict, turn_dict = compute_turn_decisions(hyp_arr_simple)
    scorer_simple = ScoreResult(true_dict, pred_dict, turn_dict, labels)
    assert (
        scorer_simple.true_arr_hard_label == ["NA", "T", "C", "C", "NA", "C", "BC"]
    ).all()
    # based on LabelThreshold
    assert (scorer_simple.pred_arr == ["NA", "T", "C", "IN", "T", "C", "BC"]).all()
    gt_F1 = {}
    # For C, TP= 2, FP=0, FN=1, TN=4
    # For positive class, Prec= 1.0, Rec= 0.67, F1 = 0.8
    # For  negative class, Prec= 0.8, Rec = 1.0, F1 = 0.89
    # Hence macro f1 = 0.844
    gt_F1["C"] = 0.844
    # For NA, TP= 1, FP=0, FN=1, TN=5
    # For positive class, Prec= 1.0, Rec= 0.5, F1= 0.67
    # For  negative class, Prec= 0.83, Rec= 1.0, F1= 0.91
    # Hence macro f1 = 0.788
    gt_F1["NA"] = 0.788
    # For I, TP= 0, FP=1, FN=0, TN=6
    # For positive class, F1= 0.0 (no positive instance)
    # For  negative class, Prec = 1.00, Rec= 0.86, F1= 0.92
    # Hence macro f1 = 0.462
    gt_F1["IN"] = 0.462
    # For IN, TP= 1, FP=0, FN=0, TN=6
    # For positive class, Prec =1.0, Rec =1.0, F1= 1.0
    # For  negative class, Prec =1.0, Rec =1.0, F1= 1.0
    # Hence macro f1 = 1.0
    gt_F1["BC"] = 1.0
    # For T, TP= 1, FP=1, FN=0, TN=5
    # For positive class, Prec =0.5, Rec =1.0, F1= 0.67
    # For  negative class, Prec =1.0, Rec =0.83, F1= 0.67
    # Hence macro f1 = 0.788
    gt_F1["T"] = 0.788
    assert scorer_simple.compute_F1() == gt_F1


def test_confusion_matrix_simple():
    ref_arr_simple = [
        "moshi_audio_1\
        0.13,0.59,0.02,0.02,0.25\
        0.06,0.04,0.04,0.07,0.78\
        0.52,0.25,0.03,0.04,0.16\
        0.67,0.10,0.01,0.01,0.21\
        0.20,0.62,0.02,0.02,0.14\
        0.84,0.01,0.13,0.02,0.01\
        0.13,0.30,0.04,0.41,0.12\
        "
    ]
    hyp_arr_simple = [
        "moshi_audio_1,0.2,0.24,NA,NA",
        "moshi_audio_1,0.24,0.28,T,NA",
        "moshi_audio_1,0.28,0.32,C,A",
        "moshi_audio_1,0.32,0.36,IN,A",
        "moshi_audio_1,0.36,0.40,T,A",
        "moshi_audio_1,0.40,0.44,C,B",
        "moshi_audio_1,0.44,0.48,BC,B",
    ]
    true_dict = compute_turn_likelihoods(
        ref_arr_simple, ModelParam.min_start_time.value, ModelParam.chunk_length.value
    )
    pred_dict, turn_dict = compute_turn_decisions(hyp_arr_simple)
    scorer_simple = ScoreResult(true_dict, pred_dict, turn_dict, labels)
    # Hard labels are same as at line 98, using that we get the confusion matrix below
    gt_array = [
        [2, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    res = scorer_simple.compute_confusion_matrix() == gt_array
    assert res.all()


def test_turn_change_metric_simple():
    ref_arr_simple = [
        "moshi_audio_1\
        0.13,0.59,0.02,0.02,0.25\
        0.06,0.04,0.04,0.07,0.78\
        0.52,0.25,0.03,0.04,0.16\
        0.67,0.10,0.01,0.01,0.21\
        0.20,0.62,0.02,0.02,0.14\
        0.84,0.01,0.13,0.02,0.01\
        0.13,0.30,0.04,0.41,0.12\
        0.13,0.30,0.04,0.12,0.41\
        "
    ]
    hyp_arr_simple = [
        "moshi_audio_1,0.2,0.24,NA,B",
        "moshi_audio_1,0.24,0.28,NA,B",
        "moshi_audio_1,0.28,0.32,NA,B",
        "moshi_audio_1,0.32,0.36,C,B",
        "moshi_audio_1,0.36,0.40,NA,B",
        "moshi_audio_1,0.40,0.44,NA,B",
        "moshi_audio_1,0.44,0.48,NA,B",
        "moshi_audio_1,0.48,0.52,T,B",
    ]
    # To calculate this metric we look at cases where user is speaking but pauses
    # i.e. line 4 and line 8 in hyp_arr_simple
    # For accuracy_pause, line 4 is where system let user continue.
    # Where (Turn Change (0.21) - Continuation (0.67) Likelihood)
    #  < MetricThreshold.turn_change.value
    # Thus Judge Label is Continuation, hence accuracy_pause = 100.0
    # For accuracy_turn_change, line 8 is where system decides to speak up.
    # Where Turn Change (0.41) - Continuation (0.13) Likelihood
    # > MetricThreshold.turn_change.value
    # Thus Judge Label is Turn Change, hence accuracy_turn_change = 100.0
    true_dict = compute_turn_likelihoods(
        ref_arr_simple, ModelParam.min_start_time.value, ModelParam.chunk_length.value
    )
    pred_dict, turn_dict = compute_turn_decisions(hyp_arr_simple)
    scorer_simple = ScoreResult(true_dict, pred_dict, turn_dict, labels)
    assert scorer_simple.turn_change_metric() == (100.0, 100.0)


def test_make_backchannel_metric_simple():
    ref_arr_simple = [
        "moshi_audio_1\
        0.13,0.59,0.02,0.02,0.25\
        0.13,0.30,0.04,0.12,0.41\
        0.52,0.25,0.03,0.04,0.16\
        0.67,0.10,0.01,0.01,0.21\
        0.20,0.62,0.02,0.02,0.14\
        0.84,0.01,0.13,0.02,0.01\
        0.13,0.30,0.04,0.41,0.12\
        0.06,0.04,0.04,0.07,0.78\
        "
    ]
    hyp_arr_simple = [
        "moshi_audio_1,0.2,0.24,NA,A",
        "moshi_audio_1,0.24,0.28,T,A",
        "moshi_audio_1,0.28,0.32,C,B",
        "moshi_audio_1,0.32,0.36,C,B",
        "moshi_audio_1,0.36,0.40,BC,B",
        "moshi_audio_1,0.40,0.44,NA,B",
        "moshi_audio_1,0.44,0.48,BC,B",
        "moshi_audio_1,0.48,0.52,C,B",
    ]
    # To calculate this metric we look at cases where user is speaking
    # and system has not backchannel yet
    # For accuracy_backchannel, line 5 and 7 in hyp_arr_simple is
    # where system starts to backchannel.
    # For line 5, BackChannel Likelihood (0.02) < MetricThreshold.backchannel.value
    # , thus Judge Label is no backchannel
    # For line 7, BackChannel Likelihood (0.41) > MetricThreshold.backchannel.value
    # , thus Judge Label is backchannel
    # Hence accuracy_backchannel = 50.0
    # For accuracy_no_backchannel, line 3 and 4 in hyp_arr_simple is
    # where system decides to not backchannel
    # Where BackChannel Likelihood (0.04, 0.01)
    # < MetricThreshold.backchannel.value,
    # Thus Judge Label is no backchannel for both cases,
    # hence accuracy_no_backchannel = 100.0
    true_dict = compute_turn_likelihoods(
        ref_arr_simple, ModelParam.min_start_time.value, ModelParam.chunk_length.value
    )
    pred_dict, turn_dict = compute_turn_decisions(hyp_arr_simple)
    scorer_simple = ScoreResult(true_dict, pred_dict, turn_dict, labels)
    assert scorer_simple.make_backchannel_metric() == (50.0, 100.0)


def test_make_interruption_metric_simple():
    ref_arr_simple = [
        "moshi_audio_1\
        0.13,0.59,0.02,0.02,0.25\
        0.13,0.30,0.04,0.12,0.41\
        0.52,0.25,0.03,0.04,0.16\
        0.67,0.10,0.01,0.01,0.21\
        0.20,0.02,0.62,0.02,0.14\
        0.84,0.01,0.13,0.02,0.01\
        0.41,0.13,0.04,0.30,0.12\
        0.06,0.04,0.78,0.07,0.04\
        "
    ]
    hyp_arr_simple = [
        "moshi_audio_1,0.2,0.24,NA,A",
        "moshi_audio_1,0.24,0.28,T,A",
        "moshi_audio_1,0.28,0.32,C,B",
        "moshi_audio_1,0.32,0.36,C,B",
        "moshi_audio_1,0.36,0.40,C,B",
        "moshi_audio_1,0.40,0.44,NA,B",
        "moshi_audio_1,0.44,0.48,C,B",
        "moshi_audio_1,0.48,0.52,IN,B",
    ]

    # To calculate this metric we look at cases where user is speaking
    # and system has not interrupted yet
    # For accuracy_interrupt, line 8 in hyp_arr_simple
    # is where system starts to interrupt.
    # For line 8, Interruption Likelihood (0.78) - Continuation Likelihood (0.06)
    # > MetricThreshold.interrupt.value
    # thus Judge Label is interruption, Hence accuracy_interrupt = 100.0
    # For accuracy_no_interrupt, line 4 and 5 in hyp_arr_simple is
    # where system decides to not interrupt
    # For line 4, Interruption Likelihood (0.01) - Continuation Likelihood (0.67)
    # < MetricThreshold.interrupt.value, thus Judge Label is continuation
    # For line 5, Interruption Likelihood (0.62) - Continuation Likelihood (0.20)
    # > MetricThreshold.interrupt.value, thus Judge Label is interruption
    # Hence accuracy_no_interrupt = 50.0
    true_dict = compute_turn_likelihoods(
        ref_arr_simple, ModelParam.min_start_time.value, ModelParam.chunk_length.value
    )
    pred_dict, turn_dict = compute_turn_decisions(hyp_arr_simple)
    scorer_simple = ScoreResult(true_dict, pred_dict, turn_dict, labels)
    assert scorer_simple.make_interruption_metric() == (100.0, 50.0)


def test_turn_willingness_metric_simple():
    ref_arr_simple = [
        "moshi_audio_1\
        0.13,0.59,0.02,0.02,0.25\
        0.06,0.04,0.04,0.07,0.78\
        0.52,0.25,0.03,0.04,0.16\
        0.67,0.10,0.01,0.01,0.21\
        0.20,0.62,0.02,0.02,0.14\
        0.84,0.01,0.13,0.02,0.01\
        0.13,0.30,0.04,0.41,0.12\
        0.33,0.30,0.04,0.12,0.21\
        "
    ]
    hyp_arr_simple = [
        "moshi_audio_1,0.2,0.24,NA,A",
        "moshi_audio_1,0.24,0.28,NA,A",
        "moshi_audio_1,0.28,0.32,NA,A",
        "moshi_audio_1,0.32,0.36,C,A",
        "moshi_audio_1,0.36,0.40,NA,A",
        "moshi_audio_1,0.40,0.44,NA,A",
        "moshi_audio_1,0.44,0.48,NA,A",
        "moshi_audio_1,0.48,0.52,T,A",
    ]
    # To calculate this metric we look at cases where system is speaking
    # but pauses i.e. line 4 and line 8 in hyp_arr_simple
    # For accuracy_pause, line 4 is where system continues.
    # Where Turn Change Likelihood (0.21) - Continuation Likelihood (0.67)
    # < MetricThreshold.turn_change.value
    # Thus Judge Label is Continuation, hence accuracy_pause = 100.0
    # For accuracy_turn_change, line 8 is where system decides to speak up.
    # Where Turn Change Likelihood (0.21) - Continuation Likelihood (0.33)
    # < MetricThreshold.turn_change.value
    # Thus Judge Label is Continuation, hence accuracy_turn_change = 0.0
    true_dict = compute_turn_likelihoods(
        ref_arr_simple, ModelParam.min_start_time.value, ModelParam.chunk_length.value
    )
    pred_dict, turn_dict = compute_turn_decisions(hyp_arr_simple)
    scorer_simple = ScoreResult(true_dict, pred_dict, turn_dict, labels)
    assert scorer_simple.turn_willingness_metric() == (100.0, 0.0)


def test_handle_interruption_metric_simple():
    ref_arr_simple = [
        "moshi_audio_1\
        0.13,0.59,0.02,0.02,0.25\
        0.06,0.04,0.04,0.07,0.78\
        0.52,0.25,0.03,0.04,0.16\
        0.67,0.10,0.01,0.01,0.21\
        0.20,0.62,0.02,0.02,0.14\
        0.01,0.01,0.13,0.02,0.84\
        0.13,0.30,0.04,0.41,0.12\
        0.33,0.30,0.04,0.12,0.21\
        "
    ]
    hyp_arr_simple = [
        "moshi_audio_1,0.2,0.24,IN,A",
        "moshi_audio_1,0.24,0.28,IN,AB",
        "moshi_audio_1,0.28,0.32,C,AB",
        "moshi_audio_1,0.32,0.36,C,A",
        "moshi_audio_1,0.36,0.40,IN,A",
        "moshi_audio_1,0.40,0.44,T,AB",
        "moshi_audio_1,0.44,0.48,IN,B",
        "moshi_audio_1,0.48,0.52,T,BA",
    ]
    # To calculate this metric we look at cases where system is speaking
    # and user interrupts (speaker turn = "AB") i.e. line 3 and line 6 in hyp_arr_simple
    # For accuracy_unsuccess, line 3 is where user cannot successfully interrupt.
    # Where Turn Change Likelihood (0.16) - Continuation Likelihood (0.52)
    # < MetricThreshold.success_interrupt.value
    # Thus Judge Label is Continuation, hence accuracy_unsuccess = 100.0
    # For accuracy_success, line 6 is where system decides to speak up.
    # Where Turn Change Likelihood (0.84) - Continuation Likelihood (0.01)
    # > MetricThreshold.success_interrupt.value
    # Thus Judge Label is Turn Change, hence accuracy_success = 100.0
    true_dict = compute_turn_likelihoods(
        ref_arr_simple, ModelParam.min_start_time.value, ModelParam.chunk_length.value
    )
    pred_dict, turn_dict = compute_turn_decisions(hyp_arr_simple)
    scorer_simple = ScoreResult(true_dict, pred_dict, turn_dict, labels)
    assert scorer_simple.handle_interruption_metric() == (100.0, 100.0)
