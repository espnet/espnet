// nnet2/train-nnet-ensemble.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//           2014  Xiaohui Zhang

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

#ifndef KALDI_NNET2_TRAIN_NNET_ENSEMBLE_H_
#define KALDI_NNET2_TRAIN_NNET_ENSEMBLE_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet2 {


struct NnetEnsembleTrainerConfig {
  int32 minibatch_size;
  int32 minibatches_per_phase;
  double beta;

  NnetEnsembleTrainerConfig(): minibatch_size(500),
                             minibatches_per_phase(50),
                             beta(0.5) { }
  
  void Register (OptionsItf *opts) {
    opts->Register("minibatch-size", &minibatch_size,
                   "Number of samples per minibatch of training data.");
    opts->Register("minibatches-per-phase", &minibatches_per_phase,
                   "Number of minibatches to wait before printing training-set "
                   "objective.");
    opts->Register("beta", &beta, 
                   "weight of the second term in the objf, which is the cross-entropy "
                   "between the output posteriors and the averaged posteriors from other nets.");
  }  
};


// Similar as NnetTrainer, Class NnetEnsembleTrainer first batches
// up the input into minibatches and feed the data into every nnet in 
// the ensemble, call Propogate to do forward propogation, and 
// collect the output posteriors. The posteriors from different 
// nets are averaged and then used to compute the additional term 
// in the objf: (a constant times) the cross-entropy between each 
// net's output posteriors and the averaged posteriors of 
// the whole nnet ensemble. We also calculate the derivs and 
// then call Backprop() to update each net separately.

class NnetEnsembleTrainer {
 public:
  NnetEnsembleTrainer(const NnetEnsembleTrainerConfig &config,
                      std::vector<Nnet*> nnet_ensemble);
  
  /// TrainOnExample will take the example and add it to a buffer;
  /// if we've reached the minibatch size it will do the training.
  void TrainOnExample(const NnetExample &value);

  ~NnetEnsembleTrainer();
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetEnsembleTrainer);
  
  void TrainOneMinibatch();
  
  // The following function is called by TrainOneMinibatch()
  // when we enter a new phase.
  void BeginNewPhase(bool first_time);
  
  // Things we were given in the initializer:
  NnetEnsembleTrainerConfig config_;

  std::vector<Nnet*> nnet_ensemble_; // the nnet ensemble we're training.
  std::vector<NnetUpdater*> updater_ensemble_;

  // State information:
  int32 num_phases_;
  int32 minibatches_seen_this_phase_;
  std::vector<NnetExample> buffer_; 

  // ratio of the supervision, when interpolating the supervision with the averaged posteriors. 
  double beta_;
  double avg_logprob_this_phase_; // Needed for accumulating train log-prob on each phase.
  double count_this_phase_; // count corresponding to the above.
};



} // namespace nnet2
} // namespace kaldi

#endif
