// nnetbin/rbm-train-cd1-frmshuff.cc

// Copyright 2012-2013  Brno University of Technology (Author: Karel Vesely)

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

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-rbm.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-rand.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Train RBM by Contrastive Divergence alg. with 1 step of "
      "Markov Chain Monte-Carlo.\n"
      "The tool can perform several iterations (--num-iters) "
      "or it can subsample the training dataset (--drop-data)\n"

      "Usage: rbm-train-cd1-frmshuff [options] <model-in> "
      "<feature-rspecifier> <model-out>\n"
      "e.g.: rbm-train-cd1-frmshuff 1.rbm.init scp:train.scp 1.rbm\n";

    ParseOptions po(usage);

    RbmTrainOptions trn_opts, trn_opts_rbm;
    trn_opts.Register(&po);
    LossOptions loss_opts;
    loss_opts.Register(&po);

    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    bool with_bug = true;
    po.Register("with-bug", &with_bug,
        "Apply bug which led to better results (set-initial-momentum-to-max)");

    int32 num_iters = 1;
    po.Register("num-iters", &num_iters,
                "Number of iterations (smaller datasets should have more iterations, "
                "iterating within tool because of linear momentum scheduling)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in 'nnet1' format");

    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.minibatch_size = 100;
    rnd_opts.Register(&po);

    kaldi::int32 max_frames = 6000;
    po.Register("max-frames", &max_frames,
        "Maximum number of frames an utterance can have (skipped if longer)");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2);

    std::string target_model_filename;
    target_model_filename = po.GetArg(3);


    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet rbm_transf;
    if (feature_transform != "") {
      rbm_transf.Read(feature_transform);
    }

    // Read nnet, extract the RBM,
    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.NumComponents() == 1);
    KALDI_ASSERT(nnet.GetComponent(0).GetType() == kaldi::nnet1::Component::kRbm);
    RbmBase &rbm = dynamic_cast<RbmBase&>(nnet.GetComponent(0));

    // Configure the RBM,
    // make some constants accessible, will use them later,
    const BaseFloat& learn_rate = trn_opts.learn_rate;
    const BaseFloat& momentum = trn_opts.momentum;
    const BaseFloat& momentum_max = trn_opts.momentum_max;
    const int32& momentum_steps = trn_opts.momentum_steps;
    const int32& momentum_step_period = trn_opts.momentum_step_period;

    // 'trn_opts_rbm' is a local copy of 'trn_opts' which is passed to RBM,
    trn_opts_rbm = trn_opts;
    // keep `effective' learning rate constant
    trn_opts_rbm.learn_rate = learn_rate * (1 - momentum);
    // pass options to RBM,
    rbm.SetRbmTrainOptions(trn_opts_rbm);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);

    CuRand<BaseFloat> cu_rand;  // parallel random number generator,
    Mse mse(loss_opts);

    CuMatrix<BaseFloat> feats_transf,
                        pos_hid, pos_hid_aux,
                        neg_vis, neg_hid;
    CuMatrix<BaseFloat> dummy_mse_mat;

    Timer time;
    KALDI_LOG << "RBM TRAINING STARTED";

    int32 iter = 1;
    KALDI_LOG << "Iteration " << iter << "/" << num_iters;

    int32 num_done = 0, num_other_error = 0;
    while (!feature_reader.Done()) {
#if HAVE_CUDA == 1
      // check that GPU is computing accurately,
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer,
      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        if (feature_randomizer.IsFull()) {
          // break the loop without calling Next(),
          // we keep the 'utt' for next round,
          break;
        }
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << "Reading " << utt;
        // get feature matrix,
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        // skip too long segments (avoid runinning out of memory)
        if (mat.NumRows() > max_frames) {
          KALDI_WARN << "Skipping " << utt
            << " that has " << mat.NumRows() << " frames,"
            << " it is longer than '--max-frames'" << max_frames;
          num_other_error++;
          continue;
        }
        // apply feature transform,
        rbm_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
        // add to randomizer,
        feature_randomizer.AddData(feats_transf);
        num_done++;

        // report the speed
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: "
            << "time elapsed = " << time_now / 60 << " min; "
            << "processed " << total_frames / time_now << " frames per sec.";
        }
      }

      // randomize,
      feature_randomizer.Randomize(
        randomizer_mask.Generate(feature_randomizer.NumFrames())
      );

      // train with data from randomizer (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next()) {
        // get the mini-batch,
        const CuMatrixBase<BaseFloat>& pos_vis = feature_randomizer.Value();
        // get the dims,
        int32 num_frames = pos_vis.NumRows(),
              dim_hid = rbm.OutputDim();
        // Create dummy frame-weights for Mse::Eval,
        Vector<BaseFloat> dummy_weights(num_frames);
        dummy_weights.Set(1.0);

        // TRAIN with CD1,
        // forward pass,
        rbm.Propagate(pos_vis, &pos_hid);

        // alter the hidden values, so we can generate negative example,
        if (rbm.HidType() == Rbm::Bernoulli) {
          pos_hid_aux.Resize(num_frames, dim_hid);
          cu_rand.BinarizeProbs(pos_hid, &pos_hid_aux);  // => 0 / 1,
        } else {
          KALDI_ASSERT(rbm.HidType() == Rbm::Gaussian);
          pos_hid_aux = pos_hid;
          cu_rand.AddGaussNoise(&pos_hid_aux);
        }

        // reconstruct pass,
        rbm.Reconstruct(pos_hid_aux, &neg_vis);
        // propagate negative examples
        rbm.Propagate(neg_vis, &neg_hid);
        // update step
        rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);
        // evaluate mean square error
        mse.Eval(dummy_weights, neg_vis, pos_vis, &dummy_mse_mat);

        total_frames += num_frames;

        // change the momentum progressively per 0.5million samples of the data
        {
          static int32 n_prev = -1;
          BaseFloat step = (momentum_max - momentum) / momentum_steps;
          // change every momentum_step_period data,
          int32 n = total_frames / momentum_step_period;
          BaseFloat momentum_actual;
          if (n > momentum_steps) {
            momentum_actual = momentum_max;
          } else {
            momentum_actual = momentum + n*step;
          }
          if (n - n_prev > 0) {
            n_prev = n;
            BaseFloat learning_rate_actual = learn_rate*(1-momentum_actual);
            KALDI_VLOG(1) << "Setting momentum "
              << (with_bug ? momentum_max : momentum_actual)
              << " and learning rate " << learning_rate_actual
              << " after processing "
              << static_cast<double>(total_frames) / 360000 << " h";
            // pass values to rbm,
            trn_opts_rbm.momentum = (with_bug ? momentum_max : momentum_actual);
            trn_opts_rbm.learn_rate = learning_rate_actual;
            rbm.SetRbmTrainOptions(trn_opts_rbm);
          }
        }
      }

      // reopen the feature stream if we will run another iteration
      if (feature_reader.Done() && (iter < num_iters)) {
        iter++;
        KALDI_LOG << "Iteration " << iter << "/" << num_iters;
        feature_reader.Close();
        feature_reader.Open(feature_rspecifier);
      }
    }

    nnet.Write(target_model_filename, binary);

    KALDI_LOG << "Done " << iter << " iterations, " << num_done << " files, "
              << "skipped " << num_other_error << " files. "
              << "[" << time.Elapsed() / 60 << " min, "
              << "processing" << total_frames / time.Elapsed() << " "
              << "frames per sec.]";

    KALDI_LOG << mse.Report();

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
