// nnet3/nnet-am-decodable-simple.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_
#define KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_

#include <vector>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"

namespace kaldi {
namespace nnet3 {


// See also the decodable object in decodable-simple-looped.h, which is better
// and faster in most situations, including TDNNs and LSTMs (but not for
// BLSTMs).


// Note: the 'simple' in the name means it applies to networks
// for which IsSimpleNnet(nnet) would return true.
struct NnetSimpleComputationOptions {
  int32 extra_left_context;
  int32 extra_right_context;
  int32 extra_left_context_initial;
  int32 extra_right_context_final;
  int32 frame_subsampling_factor;
  int32 frames_per_chunk;
  BaseFloat acoustic_scale;
  bool debug_computation;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  CachingOptimizingCompilerOptions compiler_config;

  NnetSimpleComputationOptions():
      extra_left_context(0),
      extra_right_context(0),
      extra_left_context_initial(-1),
      extra_right_context_final(-1),
      frame_subsampling_factor(1),
      frames_per_chunk(50),
      acoustic_scale(0.1),
      debug_computation(false) {
    compiler_config.cache_capacity += frames_per_chunk;
  }

  void Register(OptionsItf *opts) {
    opts->Register("extra-left-context", &extra_left_context,
                   "Number of frames of additional left-context to add on top "
                   "of the neural net's inherent left context (may be useful in "
                   "recurrent setups");
    opts->Register("extra-right-context", &extra_right_context,
                   "Number of frames of additional right-context to add on top "
                   "of the neural net's inherent right context (may be useful in "
                   "recurrent setups");
    opts->Register("extra-left-context-initial", &extra_left_context_initial,
                   "If >= 0, overrides the --extra-left-context value at the "
                   "start of an utterance.");
    opts->Register("extra-right-context-final", &extra_right_context_final,
                   "If >= 0, overrides the --extra-right-context value at the "
                   "end of an utterance.");
    opts->Register("frame-subsampling-factor", &frame_subsampling_factor,
                   "Required if the frame-rate of the output (e.g. in 'chain' "
                   "models) is less than the frame-rate of the original "
                   "alignment.");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic log-likelihoods (caution: is a no-op "
                   "if set in the program nnet3-compute");
    opts->Register("frames-per-chunk", &frames_per_chunk,
                   "Number of frames in each chunk that is separately evaluated "
                   "by the neural net.  Measured before any subsampling, if the "
                   "--frame-subsampling-factor options is used (i.e. counts "
                   "input frames");
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);

    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};

/*
  This class handles the neural net computation; it's mostly accessed
  via other wrapper classes.

  Note: this class used to be called NnetDecodableBase.

  It can accept just input features, or input features plus iVectors.  */
class DecodableNnetSimple {
 public:
  /**
     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] opts   The options class.  Warning: it includes an acoustic
                        weight, whose default is 0.1; you may sometimes want to
                        change this to 1.0.
     @param [in] nnet   The neural net that we're going to do the computation with
     @param [in] priors Vector of priors-- if supplied and nonempty, we subtract
                        the log of these priors from the nnet output.
     @param [in] feats  The input feature matrix.
     @param [in] compiler  A pointer to the compiler object to use-- this enables the
                        user to maintain a common object in the calling code that
                        will cache computations across decodes.  Note: the compiler code
                        has no locking mechanism (and it would be tricky to design one,
                        as we'd need to lock the individual computations also),
                        so the calling code has to make sure that if there are
                        multiple threads, they do not share the same compiler
                        object.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] online_ivectors
                        If you are using iVectors estimated 'online'
                        a pointer to the iVectors, else NULL.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
  */
  DecodableNnetSimple(const NnetSimpleComputationOptions &opts,
                      const Nnet &nnet,
                      const VectorBase<BaseFloat> &priors,
                      const MatrixBase<BaseFloat> &feats,
                      CachingOptimizingCompiler *compiler,
                      const VectorBase<BaseFloat> *ivector = NULL,
                      const MatrixBase<BaseFloat> *online_ivectors = NULL,
                      int32 online_ivector_period = 1);


  // returns the number of frames of likelihoods.  The same as feats_.NumRows()
  // in the normal case (but may be less if opts_.frame_subsampling_factor !=
  // 1).
  inline int32 NumFrames() const { return num_subsampled_frames_; }

  inline int32 OutputDim() const { return output_dim_; }

  // Gets the output for a particular frame, with 0 <= frame < NumFrames().
  // 'output' must be correctly sized (with dimension OutputDim()).
  void GetOutputForFrame(int32 frame, VectorBase<BaseFloat> *output);

  // Gets the output for a particular frame and pdf_id, with
  // 0 <= subsampled_frame < NumFrames(),
  // and 0 <= pdf_id < OutputDim().
  inline BaseFloat GetOutput(int32 subsampled_frame, int32 pdf_id) {
    if (subsampled_frame < current_log_post_subsampled_offset_ ||
        subsampled_frame >= current_log_post_subsampled_offset_ +
                            current_log_post_.NumRows())
      EnsureFrameIsComputed(subsampled_frame);
    return current_log_post_(subsampled_frame -
                             current_log_post_subsampled_offset_,
                             pdf_id);
  }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableNnetSimple);

  // This call is made to ensure that we have the log-probs for this frame
  // cached in current_log_post_.
  void EnsureFrameIsComputed(int32 subsampled_frame);

  // This function does the actual nnet computation; it is called from
  // EnsureFrameIsComputed.  Any padding at file start/end is done by
  // the caller of this function (so the input should exceed the output
  // by a suitable amount of context).  It puts its output in current_log_post_.
  void DoNnetComputation(int32 input_t_start,
                         const MatrixBase<BaseFloat> &input_feats,
                         const VectorBase<BaseFloat> &ivector,
                         int32 output_t_start,
                         int32 num_subsampled_frames);

  // Gets the iVector that will be used for this chunk of frames, if we are
  // using iVectors (else does nothing).  note: the num_output_frames is
  // interpreted as the number of t value, which in the subsampled case is not
  // the same as the number of subsampled frames (it would be larger by
  // opts_.frame_subsampling_factor).
  void GetCurrentIvector(int32 output_t_start,
                         int32 num_output_frames,
                         Vector<BaseFloat> *ivector);

  // called from constructor
  void CheckAndFixConfigs();

  // returns dimension of the provided iVectors if supplied, or 0 otherwise.
  int32 GetIvectorDim() const;

  NnetSimpleComputationOptions opts_;
  const Nnet &nnet_;
  int32 nnet_left_context_;
  int32 nnet_right_context_;
  int32 output_dim_;
  // the log priors (or the empty vector if the priors are not set in the model)
  CuVector<BaseFloat> log_priors_;
  const MatrixBase<BaseFloat> &feats_;
  // note: num_subsampled_frames_ will equal feats_.NumRows() in the normal case
  // when opts_.frame_subsampling_factor == 1.
  int32 num_subsampled_frames_;

  // ivector_ is the iVector if we're using iVectors that are estimated in batch
  // mode.
  const VectorBase<BaseFloat> *ivector_;

  // online_ivector_feats_ is the iVectors if we're using online-estimated ones.
  const MatrixBase<BaseFloat> *online_ivector_feats_;
  // online_ivector_period_ helps us interpret online_ivector_feats_; it's the
  // number of frames the rows of ivector_feats are separated by.
  int32 online_ivector_period_;

  // a reference to a compiler passed in via the constructor, which may be
  // declared at the top level of the program so that we don't have to recompile
  // computations each time.
  CachingOptimizingCompiler &compiler_;

  // The current log-posteriors that we got from the last time we
  // ran the computation.
  Matrix<BaseFloat> current_log_post_;
  // The time-offset of the current log-posteriors.  Note: if
  // opts_.frame_subsampling_factor > 1, this will be measured in subsampled
  // frames.
  int32 current_log_post_subsampled_offset_;
};

class DecodableAmNnetSimple: public DecodableInterface {
 public:
  /**
     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] opts   The options class.  Warning: it includes an acoustic
                        weight, whose default is 0.1; you may sometimes want to
                        change this to 1.0.
     @param [in] trans_model  The transition model to use.  This takes care of the
                        mapping from transition-id (which is an arg to
                        LogLikelihood()) to pdf-id (which is used internally).
     @param [in] am_nnet   The neural net that we're going to do the computation with;
                         we also get the priors to divide by, if applicable, from here.
     @param [in] feats   A pointer to the input feature matrix; must be non-NULL.
                         We
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] online_ivectors
                        If you are using iVectors estimated 'online'
                        a pointer to the iVectors, else NULL.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
     @param [in,out] compiler  A pointer to a compiler [optional]-- the user
                        can declare one in the calling code and repeatedly
                        supply pointers to it, which allows for caching of computations
                        across consecutive decodes.  You'd want to have initialized
                        the compiler object with as
                        compiler(am_nnet.GetNnet(), opts.optimize_config).
  */
  DecodableAmNnetSimple(const NnetSimpleComputationOptions &opts,
                        const TransitionModel &trans_model,
                        const AmNnetSimple &am_nnet,
                        const MatrixBase<BaseFloat> &feats,
                        const VectorBase<BaseFloat> *ivector = NULL,
                        const MatrixBase<BaseFloat> *online_ivectors = NULL,
                        int32 online_ivector_period = 1,
                        CachingOptimizingCompiler *compiler = NULL);


  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual inline int32 NumFramesReady() const {
    return decodable_nnet_.NumFrames();
  }

  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetSimple);
  // This compiler object is only used if the 'compiler'
  // argument to the constructor is NULL.
  CachingOptimizingCompiler compiler_;
  DecodableNnetSimple decodable_nnet_;
  const TransitionModel &trans_model_;
};


class DecodableAmNnetSimpleParallel: public DecodableInterface {
 public:
  /**
     This decodable object is for use in multi-threaded decoding.
     It differs from DecodableAmNnetSimple in two respects:
        (1) It doesn't keep around pointers to the features and iVectors;
            instead, it creates copies of them (so the caller can
            delete the originals).
        (2) It doesn't support the user passing in a pointer to the
            CachingOptimizingCompiler-- because making that thread safe
            would be quite complicated, and in any case multi-threaded
            decoding probably makes the most sense when using CPU, and
            in that case we don't expect the compilation phase to dominate.

     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] opts   The options class.  Warning: it includes an acoustic
                        weight, whose default is 0.1; you may sometimes want to
                        change this to 1.0.
     @param [in] trans_model  The transition model to use.  This takes care of the
                        mapping from transition-id (which is an arg to
                        LogLikelihood()) to pdf-id (which is used internally).
     @param [in] am_nnet The neural net that we're going to do the computation with;
                        it may provide priors to divide by.
     @param [in] feats   A pointer to the input feature matrix; must be non-NULL.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] online_ivectors
                        If you are using iVectors estimated 'online'
                        a pointer to the iVectors, else NULL.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
  */
  DecodableAmNnetSimpleParallel(
      const NnetSimpleComputationOptions &opts,
      const TransitionModel &trans_model,
      const AmNnetSimple &am_nnet,
      const MatrixBase<BaseFloat> &feats,
      const VectorBase<BaseFloat> *ivector = NULL,
      const MatrixBase<BaseFloat> *online_ivectors = NULL,
      int32 online_ivector_period = 1);


  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual inline int32 NumFramesReady() const {
    return decodable_nnet_->NumFrames();
  }

  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

  ~DecodableAmNnetSimpleParallel() { DeletePointers(); }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetSimpleParallel);
  void DeletePointers();

  CachingOptimizingCompiler compiler_;
  const TransitionModel &trans_model_;

  Matrix<BaseFloat> *feats_copy_;
  Vector<BaseFloat> *ivector_copy_;
  Matrix<BaseFloat> *online_ivectors_copy_;

  DecodableNnetSimple *decodable_nnet_;
};



} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_
