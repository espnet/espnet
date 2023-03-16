import argparse
import glob
import json
import os
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import jiwer
import pandas as pd
from pyannote.core.utils.types import Label
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.metrics.matcher import HungarianMapper
from pyannote.metrics.segmentation import Annotation, Segment, Timeline
from pyannote.metrics.types import MetricComponents
from tabulate import tabulate
from tqdm import tqdm


def compute_der(df_or_dict):
    if isinstance(df_or_dict, dict):
        der = (
            df_or_dict["false alarm"]
            + df_or_dict["missed detection"]
            + df_or_dict["confusion"]
        ) / df_or_dict["total"]
    elif isinstance(df_or_dict, pd.DataFrame):
        der = (
            df_or_dict["false alarm"].sum()
            + df_or_dict["missed detection"].sum()
            + df_or_dict["confusion"].sum()
        ) / df_or_dict["total"].sum()
    else:
        raise NotImplementedError

    return der


def compute_jer(df_or_dict):
    if isinstance(df_or_dict, dict):
        jer = df_or_dict["speaker error"] / df_or_dict["speaker count"]
    elif isinstance(df_or_dict, pd.DataFrame):
        jer = df_or_dict["speaker error"].sum() / df_or_dict["speaker count"].sum()
    else:
        raise NotImplementedError

    return jer


class DERComputer(IdentificationErrorRate):
    """Modified from https://github.com/pyannote/
    pyannote-metrics/blob/14af03ca61527621cfc0a3ed7237cc2969681915/
    pyannote/metrics/diarization.py"""

    def __init__(self, collar: float = 0.0, skip_overlap: bool = False, **kwargs):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = HungarianMapper()

    def get_optimal_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
    ) -> Dict[Label, Label]:
        mapping = self.mapper_(hypothesis, reference)
        mapped = hypothesis.rename_labels(mapping=mapping)
        return mapped, mapping

    def get_uemified(self, reference, hypothesis, uem: Optional[Timeline] = None):
        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        return reference, hypothesis, uem

    def compute_der(
        self,
        reference: Annotation,
        mapped: Annotation,
        uem: Optional[Timeline] = None,
        **kwargs
    ):
        components = super(DERComputer, self).compute_components(
            reference, mapped, uem=uem, collar=0.0, skip_overlap=False, **kwargs
        )
        der = compute_der(components)
        components.update({"diarization error rate": der})
        return components


class JERComputer(DiarizationErrorRate):
    """Modified from https://github.com/pyannote/
    pyannote-metrics/blob/14af03ca61527621cfc0a3ed7237cc2969681915/
    pyannote/metrics/diarization.py"""

    def __init__(self, collar=0.0, skip_overlap=False, **kwargs):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = HungarianMapper()

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "speaker count",
            "speaker error",
        ]

    def compute_jer(self, reference, hypothesis, mapping):
        detail = self.init_components()
        for ref_speaker in reference.labels():
            hyp_speaker = mapping.get(ref_speaker, None)
            if hyp_speaker is None:
                jer = 1.0
            else:
                r = reference.label_timeline(ref_speaker)
                h = hypothesis.label_timeline(hyp_speaker)
                total = r.union(h).support().duration()
                fa = h.duration() - h.crop(r).duration()
                miss = r.duration() - r.crop(h).duration()
                jer = (fa + miss) / total
            detail["speaker count"] += 1
            detail["speaker error"] += jer

        jer = compute_jer(detail)
        detail.update({"Jaccard error rate": jer})
        return detail


def compute_wer(df_or_dict):
    if isinstance(df_or_dict, dict):
        wer = (
            df_or_dict["substitutions"]
            + df_or_dict["deletions"]
            + df_or_dict["insertions"]
        ) / (df_or_dict["substitutions"] + df_or_dict["deletions"] + df_or_dict["hits"])
    elif isinstance(df_or_dict, pd.DataFrame):
        wer = (
            df_or_dict["substitutions"].sum()
            + df_or_dict["deletions"].sum()
            + df_or_dict["insertions"].sum()
        ) / (
            df_or_dict["substitutions"].sum()
            + df_or_dict["deletions"].sum()
            + df_or_dict["hits"].sum()
        )
    else:
        raise NotImplementedError

    return wer


def compute_diar_errors(hyp_segs, ref_segs, uem_boundaries=None, collar=0.5):
    # computing all diarization errors for each session here.
    # find optimal mapping too, which will then be used to find the WER.
    if uem_boundaries is not None:
        uem = Timeline([Segment(start=uem_boundaries[0], end=uem_boundaries[-1])])
    else:
        uem = None

    def to_annotation(segs):
        out = Annotation()
        for s in segs:
            speaker = s["speaker"]
            start = float(s["start_time"])
            end = float(s["end_time"])
            out[Segment(start, end)] = speaker
        return out

    hyp_annotation = to_annotation(hyp_segs)
    ref_annotation = to_annotation(ref_segs)

    der_computer = DERComputer(collar=collar, skip_overlap=False)
    reference, hypothesis, uem = der_computer.get_uemified(
        ref_annotation, hyp_annotation, uem=uem
    )
    mapped, mapping = der_computer.get_optimal_mapping(reference, hypothesis)
    der_score = der_computer.compute_der(reference, mapped, uem=uem)
    # avoid uemify again with custom class
    jer_compute = JERComputer(
        collar=collar, skip_overlap=False
    )  # not optimal computationally
    jer_score = jer_compute.compute_jer(reference, hypothesis, mapping)
    # error analysis here
    error_compute = IdentificationErrorAnalysis(collar=collar, skip_overlap=False)
    reference, hypothesis, errors = error_compute.difference(
        ref_annotation, hyp_annotation, uem=uem, uemified=True
    )

    return mapping, der_score, jer_score, reference, hypothesis, errors


def compute_asr_errors(hyp_segs, ref_segs, mapping=None, uem=None):
    if mapping is not None:  # using diarization
        hyp_segs_reordered = []
        for s in hyp_segs:
            new_segment = deepcopy(s)
            c_speaker = new_segment["speaker"]

            if c_speaker not in mapping.keys():
                mapping[c_speaker] = "FA_" + c_speaker  # false speaker
            new_segment["speaker"] = mapping[c_speaker]
            hyp_segs_reordered.append(new_segment)
        hyp_segs = hyp_segs_reordered

    def spk2utts(segs, uem=None):
        st_uem, end_uem = uem
        spk2utt = {k["speaker"]: [] for k in segs}
        for s in segs:
            start = float(s["start_time"])
            end = float(s["end_time"])
            # discard segments whose end is in the uem.
            if uem is not None and (end < st_uem or start > end_uem):
                continue
            spk2utt[s["speaker"]].append(s)
        return spk2utt

    hyp = spk2utts(hyp_segs, uem)
    ref = spk2utts(ref_segs, uem)

    if mapping is not None:
        # check if they have same speakers
        false_speakers = set(hyp.keys()).difference(set(ref.keys()))
        if false_speakers:
            for f_spk in list(false_speakers):
                ref[f_spk] = [{"words": "", "speaker": f_spk}]
        missed_speakers = set(ref.keys()).difference(set(hyp.keys()))
        if missed_speakers:
            for m_spk in list(missed_speakers):
                hyp[m_spk] = [{"words": "", "speaker": m_spk}]

    tot_stats = {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
    speakers_stats = []
    for spk in ref.keys():
        cat_refs = " ".join([x["words"] for x in ref[spk]])
        cat_hyps = " ".join([x["words"] for x in hyp[spk]])
        if len(cat_refs) == 0:
            # need this because jiwer cannot handle empty refs
            ldist = {
                "hits": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": len(cat_hyps.split()),
            }
        else:
            ldist = jiwer.compute_measures(cat_refs, cat_hyps)
        ldist.update(
            {
                "speaker": spk,
                "tot utterances ref": len(ref[spk]),
                "tot utterances hyp": len(hyp[spk]),
            }
        )
        speakers_stats.append(ldist)
        for k in tot_stats.keys():
            tot_stats[k] += ldist[k]

    c_wer = compute_wer(tot_stats)
    tot_stats.update({"wer": c_wer})

    return tot_stats, speakers_stats


def log_diarization(
    output_folder, scenario_tag, session, reference, hypothesis, errors
):
    """
    Logging diarization output to the specified output folder for each session.
    This is useful for analyzing errors.
    """

    sess_folder = os.path.join(output_folder, scenario_tag, session)
    Path(sess_folder).mkdir(exist_ok=True, parents=True)

    with open(os.path.join(sess_folder, "diar_errors_summary.txt"), "w") as f:
        print(errors.chart(), file=f)

    # save to disk as you may want to use them to analyze errors.
    # pyannote has some useful visualization features.
    with open(os.path.join(sess_folder, "diar_errors_pyannote.pickle"), "wb") as handle:
        pickle.dump(errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(sess_folder, "diar_ref_pyannote.pickle"), "wb") as handle:
        pickle.dump(reference, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(sess_folder, "diar_hyp_pyannote.pickle"), "wb") as handle:
        pickle.dump(hypothesis, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(sess_folder, "errors.rttm"), "w") as f:
        f.write(errors.to_rttm())
    with open(os.path.join(sess_folder, "reference.rttm"), "w") as f:
        f.write(reference.to_rttm())
    with open(os.path.join(sess_folder, "hypothesis.rttm"), "w") as f:
        f.write(hypothesis.to_rttm())


def score(
    hyp_json,
    reference_jsons,
    scenario_tag,
    output_folder,
    uem_file=None,
    collar=0.5,
    use_diarization=True,
):
    scenario_dir = os.path.join(output_folder, scenario_tag)
    Path(scenario_dir).mkdir(parents=True, exist_ok=True)

    if uem_file is not None:
        with open(uem_file, "r") as f:
            lines = f.readlines()
        lines = [x.rstrip("\n") for x in lines]
        uem2sess = {}
        for x in lines:
            sess_id, _, start, stop = x.split(" ")
            uem2sess[sess_id] = (float(start), float(stop))

    # load all reference jsons
    refs = []
    for j in reference_jsons:
        with open(j, "r") as f:
            refs.extend(json.load(f))

    with open(hyp_json, "r") as f:
        hyps = json.load(f)

    def get_sess2segs(segments):
        out = {}
        for x in segments:
            c_sess = x["session_id"]
            if c_sess not in out.keys():
                out[c_sess] = []
            out[c_sess].append(x)
        return out

    h_sess2segs = get_sess2segs(hyps)
    r_sess2segs = get_sess2segs(refs)

    if not (h_sess2segs.keys() == r_sess2segs.keys()):
        print(
            "Hypothesis JSON does not have all sessions as in the reference JSONs."
            "The sessions that are missing: {}".format(
                set(h_sess2segs.keys()).difference(set(r_sess2segs.keys()))
            )
        )
        raise RuntimeError

    all_sess_stats = []
    all_spk_stats = []
    sessions = list(r_sess2segs.keys())
    for indx in tqdm(range(len(sessions))):
        session = sessions[indx]
        hyp_segs = sorted(h_sess2segs[session], key=lambda x: float(x["start_time"]))
        ref_segs = sorted(r_sess2segs[session], key=lambda x: float(x["start_time"]))
        # compute diarization error and best permutation here
        if uem_file is not None:
            c_uem = uem2sess[session]
        else:
            c_uem = None

        sess_dir = os.path.join(scenario_dir, session)
        Path(sess_dir).mkdir(exist_ok=True)

        if use_diarization:
            (
                mapping,
                der_score,
                jer_score,
                reference,
                hypothesis,
                errors,
            ) = compute_diar_errors(hyp_segs, ref_segs, c_uem, collar=collar)

            log_diarization(
                output_folder, scenario_tag, session, reference, hypothesis, errors
            )
            # save ref hyps and errors in a folder
            # save also compatible format for audacity
            c_sess_stats = {
                "session id": session,
                "scenario": scenario_tag,
                "num spk hyp": len(hypothesis.labels()),
                "num spk ref": len(reference.labels()),
                "tot utterances hyp": len(hypothesis),
                "tot utterances ref": len(reference),
            }

            c_sess_stats.update({k: v for k, v in der_score.items()})
            c_sess_stats.update({k: v for k, v in jer_score.items()})
            asr_err_sess, asr_err_spk = compute_asr_errors(
                hyp_segs, ref_segs, mapping, uem=c_uem
            )

        else:
            if not len(hyp_segs) == len(ref_segs):
                warnings.warn(
                    "If oracle diarization was used, "
                    "I expect the hypothesis to have the same number "
                    "of utterances as the "
                    "reference. Have you discarded some utterances "
                    "(e.g. too long) ? "
                    "These will be counted as deletions so be careful !"
                )
            asr_err_sess, asr_err_spk = compute_asr_errors(
                hyp_segs, ref_segs, uem=c_uem
            )
            c_sess_stats = {
                "session id": session,
                "scenario": scenario_tag,
                "num spk hyp": len(set([x["speaker"] for x in hyp_segs])),
                "num spk ref": len(set([x["speaker"] for x in ref_segs])),
                "tot utterances hyp": len(hyp_segs),
                "tot utterances ref": len(ref_segs),
            }
            # add to each speaker, session id and scenario
        [
            x.update({"session_id": session, "scenario": scenario_tag})
            for x in asr_err_spk
        ]
        c_sess_stats.update(asr_err_sess)
        all_sess_stats.append(c_sess_stats)
        all_spk_stats.extend(asr_err_spk)

    sess_df = pd.DataFrame(all_sess_stats)
    # pretty print because it may be useful
    print(tabulate(sess_df, headers="keys", tablefmt="psql"))
    # accumulate for all scenario
    scenario_wise_df = sess_df.sum(0).to_frame().transpose()
    scenario_wise_df["scenario"] = scenario_tag
    # need to recompute these
    scenario_wer = compute_wer(sess_df)
    scenario_wise_df["wer"] = scenario_wer
    if use_diarization:
        scenario_der = compute_der(sess_df)
        scenario_wise_df["der"] = scenario_der
        scenario_jer = compute_jer(sess_df)
        scenario_wise_df["jer"] = scenario_jer
    del scenario_wise_df["session id"]  # delete session

    return scenario_wise_df, all_sess_stats, all_spk_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This script is used for scoring according to the procedure outlined in"
        " CHiME-7 DASR challenge website"
        " https://www.chimechallenge.org/current/task1/index."
        " The input is a folder containing chime6.json, mixer6.json and dipco.json"
        " for a particular partition e.g. dev set or eval. "
        "The JSON should contain predictions for all session in that partition."
        " Each JSON should contain speaker id, start, stop, session id"
        " and predicted words entries for each predicted utterance.",
        add_help=True,
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-s,--hyp_folder",
        type=str,
        help="Folder containing the JSON files relative to the system output. "
        "One file for each scenario: chime6.json, dipco.json and mixer6.json. "
        "These should contain all sessions in e.g. eval set.",
        metavar="STR",
        dest="hyp_folder",
    )

    parser.add_argument(
        "-r,--dasr_root",
        type=str,
        default="dev",
        help="Folder containing the main folder of CHiME-7 DASR dataset.",
        metavar="STR",
        dest="dasr_root",
    )

    parser.add_argument(
        "-p,--partition",
        type=str,
        default="dev",
        help="Which dataset partition is being evaluated, dev or eval.",
        metavar="STR",
        dest="partition",
    )

    parser.add_argument(
        "-o,--output_folder",
        type=str,
        metavar="STR",
        dest="output_folder",
        help="Path for the output folder where we dump all logs "
        "and useful statistics.",
    )

    parser.add_argument(
        "-d,--diarization",
        type=int,
        default=1,
        required=False,
        metavar="INT",
        dest="diarization",
        help="Whether or not use diarization to re-order the system output, if "
        "set to false we will not re-order the system output (you can set "
        "to false if you are using oracle diarization).",
    )

    parser.add_argument(
        "-f,--falign",
        type=str,
        default="",
        required=False,
        metavar="STR",
        dest="falign",
        help="Path to folder containing forced-alignment rttms. (not used now)",
    )

    parser.add_argument(
        "-c,--collar",
        type=int,
        default=500,
        required=False,
        help="Diarization metrics collar in ms. 500ms collar in pyannote "
        "is equivalent to 250ms start and end collar in dscore.",
        metavar="INT",
        dest="collar",
    )

    parser.add_argument(
        "-i,--ignore_missing",
        type=int,
        default=0,
        metavar="INT",
        required=False,
        dest="ignore_missing",
        help="If 1 will ignore missing JSON for a particular scenario, "
        "in case you want to score e.g. only DiPCo. If 0 missing the "
        "corresponding JSON for a particular scenario will raise an error.",
    )

    args = parser.parse_args()
    skip_macro = False
    spk_wise_df = []
    sess_wise_df = []
    scenario_wise_df = []
    scenarios = ["chime6", "dipco", "mixer6"]
    Path(args.output_folder).mkdir(exist_ok=True)
    for indx, scenario in enumerate(scenarios):
        hyp_json = os.path.join(args.hyp_folder, scenario + ".json")
        if bool(args.ignore_missing):
            warnings.warn(
                "I cannot find {}, so I will skip {} scenario".format(
                    hyp_json, scenario
                )
            )
            warnings.warn("Macro scores will not be computed.")
            skip_macro = True
            continue
        else:
            assert os.path.exists(hyp_json), "I cannot find {}, exiting.".format(
                hyp_json
            )
        print("###################################################")
        print("### Scoring {} Scenario ###########################".format(scenario))
        print("###################################################")
        reference_json = glob.glob(
            os.path.join(
                args.dasr_root,
                scenario,
                "transcriptions_scoring",
                args.partition,
                "*.json",
            )
        )
        uem = os.path.join(args.dasr_root, scenario, "uem", args.partition, "all.uem")
        assert (
            len(reference_json) > 0
        ), "Reference JSONS not found, is the path {} correct ?".format(
            os.path.join(
                args.dasr_root,
                scenario,
                "transcriptions_scoring",
                args.partition,
                "*.json",
            )
        )
        scenario_stats, all_sess_stats, all_spk_stats = score(
            hyp_json,
            reference_json,
            scenario,
            args.output_folder,
            uem,
            collar=float(args.collar / 1000),
            use_diarization=args.diarization,
        )
        sess_wise_df.extend(all_sess_stats)
        spk_wise_df.extend(all_spk_stats)
        scenario_wise_df.append(scenario_stats)

    sess_wise_df = pd.DataFrame(sess_wise_df)
    spk_wise_df = pd.DataFrame(spk_wise_df)
    scenario_wise_df = pd.concat(scenario_wise_df, 0)
    sess_wise_df.to_csv(os.path.join(args.output_folder, "sessions_stats.csv"))
    spk_wise_df.to_csv(os.path.join(args.output_folder, "speakers_stats.csv"))
    scenario_wise_df.to_csv(os.path.join(args.output_folder, "scenarios_stats.csv"))

    # compute scenario-wise metrics
    print("###################################################")
    print("### Metrics for all Scenarios ###")
    print("###################################################")
    print(tabulate(scenario_wise_df, headers="keys", tablefmt="psql"))
    if not skip_macro:
        print("####################################################################")
        print("### Macro-Averaged Metrics across all Scenarios (Ranking Metric) ###")
        print("####################################################################")
        macro_avg = scenario_wise_df.mean(0).to_frame().T
        macro_avg.insert(0, "scenario", "macro-average")
        print(tabulate(macro_avg, headers="keys", tablefmt="psql"))
