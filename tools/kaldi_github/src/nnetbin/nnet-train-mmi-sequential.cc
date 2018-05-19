// nnetbin/nnet-train-mmi-sequential.cc

// Copyright 2012-2016  Brno University of Technology (author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <iomanip>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-pdf-prior.h"
#include "nnet/nnet-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"


namespace kaldi {
namespace nnet1 {

void LatticeAcousticRescore(const Matrix<BaseFloat> &log_like,
                            const TransitionModel &trans_model,
                            const std::vector<int32> &state_times,
                            Lattice *lat) {
  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  KALDI_ASSERT(!state_times.empty());
  std::vector<std::vector<int32> > time_to_state(log_like.NumRows());
  for (size_t i = 0; i < state_times.size(); i++) {
    KALDI_ASSERT(state_times[i] >= 0);
    if (state_times[i] < log_like.NumRows())  // end state may be past this..
      time_to_state[state_times[i]].push_back(i);
    else
      KALDI_ASSERT(state_times[i] == log_like.NumRows()
                   && "There appears to be lattice/feature mismatch.");
  }

  for (int32 t = 0; t < log_like.NumRows(); t++) {
    for (size_t i = 0; i < time_to_state[t].size(); i++) {
      int32 state = time_to_state[t][i];
      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
           aiter.Next()) {
        LatticeArc arc = aiter.Value();
        int32 trans_id = arc.ilabel;
        if (trans_id != 0) {  // Non-epsilon input label on arc
          int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
          arc.weight.SetValue2(-log_like(t, pdf_id) + arc.weight.Value2());
          aiter.SetValue(arc);
        }
      }
    }
  }
}

}  // namespace nnet1
}  // namespace kaldi


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Perform one iteration of MMI training using SGD with per-utterance"
      "updates\n"

      "Usage:  nnet-train-mmi-sequential [options] "
      "<model-in> <transition-model-in> <feature-rspecifier> "
      "<den-lat-rspecifier> <ali-rspecifier> [<model-out>]\n"

      "e.g.: nnet-train-mmi-sequential nnet.init trans.mdl scp:feats.scp "
      "scp:denlats.scp ark:ali.ark nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.learn_rate = 0.00001;  // changing default,
    trn_opts.Register(&po);

    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in 'nnet1' format");

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    BaseFloat acoustic_scale = 1.0,
        lm_scale = 1.0,
        old_acoustic_scale = 0.0;

    po.Register("acoustic-scale", &acoustic_scale,
        "Scaling factor for acoustic likelihoods");

    po.Register("lm-scale", &lm_scale,
        "Scaling factor for \"graph costs\" (including LM costs)");

    po.Register("old-acoustic-scale", &old_acoustic_scale,
        "Add in the scores in the input lattices with this scale, "
        "rather than discarding them.");

    kaldi::int32 max_frames = 6000;
    po.Register("max-frames", &max_frames,
        "Maximum number of frames an utterance can have (skipped if longer)");

    bool drop_frames = true;
    po.Register("drop-frames", &drop_frames,
        "Drop frames, where is zero den-posterior under numerator path "
        "(ie. path not in lattice)");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        transition_model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        den_lat_rspecifier = po.GetArg(4),
        num_ali_rspecifier = po.GetArg(5),
        target_model_filename = po.GetArg(6);

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    // we will use pre-softmax activations, removing softmax,
    // - pre-softmax activations are equivalent to 'log-posterior + C_frame',
    // - all paths crossing a frame share same 'C_frame',
    // - with GMM, we also have the unnormalized acoustic likelihoods,
    if (nnet.GetLastComponent().GetType() ==
        kaldi::nnet1::Component::kSoftmax) {
      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
      nnet.RemoveLastComponent();
    } else {
      KALDI_LOG << "The nnet was without softmax. "
                << "The last component in " << model_filename << " was "
                << Component::TypeToMarker(nnet.GetLastComponent().GetType());
    }
    nnet.SetTrainOptions(trn_opts);

    // Read the class-frame-counts, compute priors,
    PdfPrior log_prior(prior_opts);

    // Read transition model,
    TransitionModel trans_model;
    ReadKaldiObject(transition_model_filename, &trans_model);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessLatticeReader den_lat_reader(den_lat_rspecifier);
    RandomAccessInt32VectorReader num_ali_reader(num_ali_rspecifier);

    CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_diff;
    Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

    if (drop_frames) {
      KALDI_LOG << "--drop-frames=true :"
                   " we will zero gradient for frames with total den/num mismatch."
                   " The mismatch is likely to be caused by missing correct path "
                   " from den-lattice due wrong annotation or search error."
                   " Leaving such frames out stabilizes the training.";
    }

    Timer time;
    double time_now = 0;
    KALDI_LOG << "TRAINING STARTED";

    int32 num_done = 0, num_no_num_ali = 0, num_no_den_lat = 0,
          num_other_error = 0, num_frm_drop = 0;

    kaldi::int64 total_frames = 0;
    double lat_like;  // total likelihood of the lattice
    double lat_ac_like;  // acoustic likelihood weighted by posterior.
    double total_mmi_obj = 0.0, mmi_obj = 0.0;
    double total_post_on_ali = 0.0, post_on_ali = 0.0;

    // main loop over utterances,
    for ( ; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!den_lat_reader.HasKey(utt)) {
        KALDI_WARN << "Missing lattice of " << utt;
        num_no_den_lat++;
        continue;
      }
      if (!num_ali_reader.HasKey(utt)) {
        KALDI_WARN << "Missing alignment of " << utt;
        num_no_num_ali++;
        continue;
      }

      // 1) get the features, numerator alignment,
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const std::vector<int32> &num_ali = num_ali_reader.Value(utt);
      // check duration of numerator alignments
      if (static_cast<int32>(num_ali.size()) != mat.NumRows()) {
        KALDI_WARN << "Duration mismatch!"
                   << " alignment " << num_ali.size()
                   << " features " << mat.NumRows();
        num_other_error++;
        continue;
      }
      if (mat.NumRows() > max_frames) {
        KALDI_WARN << "Skipping " << utt
          << " that has " << mat.NumRows() << " frames,"
          << " it is longer than '--max-frames'" << max_frames;
        num_other_error++;
        continue;
      }

      // 2) get the denominator-lattice, preprocess
      Lattice den_lat = den_lat_reader.Value(utt);
      if (den_lat.Start() == -1) {
        KALDI_WARN << "Empty lattice of " << utt << ", skipping.";
        num_other_error++;
        continue;
      }
      if (old_acoustic_scale != 1.0) {
        fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale),
                          &den_lat);
      }
      // optional sort it topologically
      kaldi::uint64 props = den_lat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&den_lat) == false) {
          KALDI_ERR << "Cycles detected in lattice.";
        }
      }
      // get the lattice length and times of states,
      std::vector<int32> state_times;
      int32 max_time = kaldi::LatticeStateTimes(den_lat, &state_times);
      // check duration of den. lattice,
      if (max_time != mat.NumRows()) {
        KALDI_WARN << "Duration mismatch!"
          << " denominator lattice " << max_time
          << " features " << mat.NumRows() << ","
          << " skipping " << utt;
        num_other_error++;
        continue;
      }

      // get dims,
      int32 num_frames = mat.NumRows(),
            num_pdfs = nnet.OutputDim();

      // 3) get the pre-softmax outputs from NN,
      // apply transform,
      nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
      // propagate through the nnet (we know it's w/o softmax),
      nnet.Propagate(feats_transf, &nnet_out);
      // subtract the log_prior,
      if (prior_opts.class_frame_counts != "") {
        log_prior.SubtractOnLogpost(&nnet_out);
      }
      // transfer it back to the host,
      nnet_out_h = Matrix<BaseFloat>(nnet_out);
      // release the buffers we don't need anymore,
      feats_transf.Resize(0, 0);
      nnet_out.Resize(0, 0);

      // 4) rescore the latice,
      LatticeAcousticRescore(nnet_out_h, trans_model, state_times, &den_lat);
      if (acoustic_scale != 1.0 || lm_scale != 1.0)
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

      // 5) get the posteriors,
      kaldi::Posterior post;
      lat_like = kaldi::LatticeForwardBackward(den_lat, &post, &lat_ac_like);

      // 6) convert the Posterior to a matrix,
      PosteriorToPdfMatrix(post, trans_model, &nnet_diff_h);

      // 7) Calculate the MMI-objective function,
      // Calculate the likelihood of correct path from acoustic score,
      // the denominator likelihood is the total likelihood of the lattice.
      double path_ac_like = 0.0;
      for (int32 t = 0; t < num_frames; t++) {
        int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
        path_ac_like += nnet_out_h(t, pdf);
      }
      path_ac_like *= acoustic_scale;
      mmi_obj = path_ac_like - lat_like;
      //
      // Note: numerator likelihood does not include graph score,
      // while denominator likelihood contains graph scores.
      // The result is offset at the MMI-objective.
      // However the offset is constant for given alignment,
      // so it does not change accross epochs.

      // Sum the den-posteriors under the correct path,
      post_on_ali = 0.0;
      for (int32 t = 0; t < num_frames; t++) {
        int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
        double posterior = nnet_diff_h(t, pdf);
        post_on_ali += posterior;
      }

      // Report,
      KALDI_VLOG(1) << "Lattice #" << num_done + 1 << " processed"
        << " (" << utt << "): found " << den_lat.NumStates()
        << " states and " << fst::NumArcs(den_lat) << " arcs.";

      KALDI_VLOG(1) << "Utterance " << utt << ": Average MMI obj. value = "
        << (mmi_obj/num_frames) << " over " << num_frames << " frames."
        << " (Avg. den-posterior on ali " << post_on_ali / num_frames << ")";


      // 7a) Search for the frames with num/den mismatch,
      int32 frm_drop = 0;
      std::vector<int32> frm_drop_vec;
      for (int32 t = 0; t < num_frames; t++) {
        int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
        double posterior = nnet_diff_h(t, pdf);
        if (posterior < 1e-20) {
          frm_drop++;
          frm_drop_vec.push_back(t);
        }
      }

      // 8) subtract the pdf-Viterbi-path,
      for (int32 t = 0; t < nnet_diff_h.NumRows(); t++) {
        int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
        nnet_diff_h(t, pdf) -= 1.0;
      }

      // 9) Drop mismatched frames from the training by zeroing the derivative,
      if (drop_frames) {
        for (int32 i = 0; i < frm_drop_vec.size(); i++) {
          nnet_diff_h.Row(frm_drop_vec[i]).Set(0.0);
        }
        num_frm_drop += frm_drop;
      }
      // Report the frame dropping
      if (frm_drop > 0) {
        std::stringstream ss;
        ss << (drop_frames?"Dropped":"[dropping disabled] Would drop")
           << " frames in " << utt << " " << frm_drop << "/" << num_frames
           << ",";
        // get frame intervals from vec frm_drop_vec,
        ss << " intervals :";
        // search for streaks of consecutive numbers,
        int32 beg_streak = frm_drop_vec[0];
        int32 len_streak = 0;
        int32 i;
        for (i = 0; i < frm_drop_vec.size(); i++, len_streak++) {
          if (beg_streak + len_streak != frm_drop_vec[i]) {
            ss << " " << beg_streak << ".." << frm_drop_vec[i-1] << "frm";
            beg_streak = frm_drop_vec[i];
            len_streak = 0;
          }
        }
        ss << " " << beg_streak << ".." << frm_drop_vec[i-1] << "frm";
        // print,
        KALDI_WARN << ss.str();
      }

      // 10) backpropagate through the nnet, update,
      nnet_diff.Resize(num_frames, num_pdfs, kUndefined);
      nnet_diff.CopyFromMat(nnet_diff_h);
      nnet.Backpropagate(nnet_diff, NULL);
      // relase the buffer, we don't need anymore,
      nnet_diff.Resize(0, 0);

      // increase time counter
      total_mmi_obj += mmi_obj;
      total_post_on_ali += post_on_ali;
      total_frames += num_frames;
      num_done++;

      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: "
          << "time elapsed = " << time_now / 60 << " min; "
          << "processed " << total_frames / time_now << " frames per sec.";
#if HAVE_CUDA == 1
        // check that GPU computes accurately,
        CuDevice::Instantiate().CheckGpuHealth();
#endif
      }

      // GRADIENT LOGGING
      // First utterance,
      if (num_done == 1) {
        KALDI_VLOG(1) << nnet.InfoPropagate();
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
      // Every 1000 utterances (--verbose=2),
      if (GetVerboseLevel() >= 2) {
        if (num_done % 1000 == 0) {
          KALDI_VLOG(2) << nnet.InfoPropagate();
          KALDI_VLOG(2) << nnet.InfoBackPropagate();
          KALDI_VLOG(2) << nnet.InfoGradient();
        }
      }
    }  // main loop over utterances,

    // After last utterance,
    KALDI_VLOG(1) << nnet.InfoPropagate();
    KALDI_VLOG(1) << nnet.InfoBackPropagate();
    KALDI_VLOG(1) << nnet.InfoGradient();

    // Add the softmax layer back before writing,
    KALDI_LOG << "Appending the softmax " << target_model_filename;
    nnet.AppendComponentPointer(new Softmax(nnet.OutputDim(), nnet.OutputDim()));
    // Store the nnet,
    nnet.Write(target_model_filename, binary);

    time_now = time.Elapsed();
    KALDI_LOG << "TRAINING FINISHED; "
              << "Time taken = " << time_now/60 << " min; processed "
              << (total_frames/time_now) << " frames per second.";

    KALDI_LOG << "Done " << num_done << " files, "
              << num_no_num_ali << " with no numerator alignments, "
              << num_no_den_lat << " with no denominator lattices, "
              << num_other_error << " with other errors.";

    KALDI_LOG << "Overall MMI-objective/frame is "
              << std::setprecision(8) << total_mmi_obj / total_frames
              << " over " << total_frames << " frames,"
              << " (average den-posterior on ali "
              << total_post_on_ali / total_frames << ","
              << " dropped " << num_frm_drop
              << " frames with num/den mismatch)";

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
