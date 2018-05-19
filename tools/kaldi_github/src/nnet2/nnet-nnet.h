// nnet2/nnet-nnet.h

// Copyright 2011-2012  Karel Vesely
//           2012-2014  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_NNET_NNET_H_
#define KALDI_NNET2_NNET_NNET_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet2/nnet-component.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>


namespace kaldi {
namespace nnet2 {


/*
  This neural net is basically a series of Components, and is a fairly
  passive object that mainly acts as a store for the Components.  Training
  is handled by a separate class NnetTrainer(), and extracting likelihoods
  for decoding is handled by DecodableNnetCpu(). 
  
  There are a couple of things that make this class more than just a vector of
  Components.

   (1) It handles frame splicing (temporal context.)
   We'd like to encompass the approach described in
   http://www.fit.vutbr.cz/research/groups/speech/publi/2011/vesely_asru2011_00042.pdf
   where at a certain point they splice together frames -10, -5, 0, +5 and +10.  It
   seems that it's not necessarily best to splice together a contiguous sequence
   of frames.

   (2) It handles situations where the input features have two parts--
   a "frame-specific" part (the normal features), and a "speaker-specific", or at
   least utterance-specific part that does not vary with the frame index.
   These features are provided separately from the frame-specific ones, to avoid
   redundancy.
*/


class Nnet {
 public:
  
  /// Returns number of components-- think of this as similar to # of layers, but
  /// e.g. the nonlinearity and the linear part count as separate components,
  /// so the number of components will be more than the number of layers.
  int32 NumComponents() const { return components_.size(); }

  const Component &GetComponent(int32 c) const;
  
  Component &GetComponent(int32 c);

  /// Sets the c'th component to "component", taking ownership of the pointer
  /// and deleting the corresponding one that we own.
  void SetComponent(int32 c, Component *component);
  
  /// Returns the left-context summed over all the Components... this is the
  /// entire left-context in frames, that the network requires.
  int32 LeftContext() const;

  /// Returns the right-context summed over all the Components... this is the
  /// entire right-context in frames, that the network requires.
  int32 RightContext() const;
  
  /// The output dimension of the network -- typically
  /// the number of pdfs.
  int32 OutputDim() const;

  /// Dimension of the input features, e.g. 13 or 40.  Does not
  /// take account of frame splicing-- that is done with the "chunk"
  /// mechanism, where you provide chunks of features over time.
  int32 InputDim() const; 
  
  /// Uses the output of the Context() functions of the network, to compute a
  /// vector of size NumComponents() + 1 indexed by component-index c, of the
  /// chunk-info at the input of each layer c, where the c+1'th element contains
  /// the chunk-info at the output of that layer.
  /// The "input_chunk_size" is the time extent of the input.  If you want to
  /// produce exactly 1 output frame per chunk, then this should equal 1 +
  /// LeftContext() + RightContext().
  void ComputeChunkInfo(int32 input_chunk_size,
                        int32 num_chunks,
                        std::vector<ChunkInfo> *chunk_info_out) const;

  void ZeroStats(); // zeroes the stats on the nonlinear layers.

  /// Copies only the statistics in layers of type NonlinearComponewnt, from
  /// this neural net, leaving everything else fixed.
  void CopyStatsFrom(const Nnet &nnet);

  /// Returns the index of the lowest-numbered component which is updatable, or
  /// NumComponents() if none are updatable.
  int32 FirstUpdatableComponent() const;
  
  /// Returns the index of the highest-numbered component which is updatable, or
  /// -1 if none are updatable.
  int32 LastUpdatableComponent() const;

  /// Returns the number of updatable components.
  int32 NumUpdatableComponents() const;
  
  /// Scales the parameters of each of the updatable components.
  /// Here, scale_params is a vector of size equal to
  /// NumUpdatableComponents()
  void ScaleComponents(const VectorBase<BaseFloat> &scales);

  /// Excise any components of type DropoutComponent or AdditiveNoiseComponent
  void RemoveDropout();

  /// Calls SetDropoutScale for all the dropout nodes.
  void SetDropoutScale(BaseFloat scale);
  
  /// Replace any components of type AffineComponentPreconditioned with
  /// components of type AffineComponent.
  void RemovePreconditioning();

  /// Replaces any components of type AffineComponent or derived classes, with
  /// components of type AffineComponentPreconditionedOnline.  E.g. rank_in =
  /// 20, rank_out = 80, num_samples_history = 2000.0, alpha = 4.0
  void SwitchToOnlinePreconditioning(int32 rank_in, int32 rank_out,
                                     int32 update_period,
                                     BaseFloat num_samples_history,
                                     BaseFloat alpha);
  
  /// For each updatatable component, adds to it
  /// the corresponding element of "other" times the
  /// appropriate element of "scales" (which has the
  /// same format as for ScaleComponents(), i.e.
  /// one entry for each updatable component).
  void AddNnet(const VectorBase<BaseFloat> &scales,
               const Nnet &other);

  /// Scales all the Components with the same scale.  This applies to
  /// UpdatableComponents, and (unlike the ScaleComponents function) to
  /// SoftmaxComponents.
  void Scale(BaseFloat scale);


  /// Adds to *this, the other neural net times the scale "alpha".  This applies
  /// to UpdatableComponents, and (unlike the other AddNnet function) to
  /// SoftmaxComponents.
  void AddNnet(BaseFloat alpha,
               const Nnet &other);

  /// Turns the last affine layer into two layers of the same type, with a
  /// smaller dimension in between-- we're keeping the top singular values of
  /// the matrix.
  void LimitRankOfLastLayer(int32 dimension);

  /// This version of AddNnet adds to *this, alpha times *other, and then scales
  /// *other by beta.  The reason why we make this a separate function is for
  /// multithreading reasons (otherwise you could do AddNnet(alpha, *iter) and then
  /// other->Scale(beta).
  void AddNnet(BaseFloat alpha,
               Nnet *other,
               BaseFloat beta);

  /// Removes final components from the neural network (used for
  /// debugging).
  void Resize(int32 num_components);


  /// Where possible, collapse multiple affine or linear components in a
  /// sequence into a single one by composing the transforms.  If
  /// match_updatableness=true, this will not collapse, say, an
  /// AffineComponent with a FixedAffineComponent or FixedLinearComponent.
  /// If false, it will collapse them.  This function won't necessarily
  /// work for all pairs of such layers.  It currently only works where
  /// one of each pair is an AffineComponent.
  void Collapse(bool match_updatableness);
  

  /// Sets the index_ values of the components.
  void SetIndexes(); 
  
  Nnet(const Nnet &other); // Copy constructor.

  Nnet(const Nnet &nnet1, const Nnet &nnet2); // Constructor that takes two
  // nnets: it concatenates the layers.
  
  Nnet() {}

  Nnet &operator = (const Nnet &other); // assignment operator.

  /// Initialize from config file.
  /// Each line of the config is either a comment line starting
  /// with whitespace then #, or it is a line that specifies one
  /// layer of the network, as accepted by Component::InitFromString().
  /// An example non-comment line is:
  /// AffineComponent learning-rate=0.01 l2-penalty=0.001 input-dim=10 output-dim=15 param-stddev=0.1
  void Init(std::istream &is);

  /// This Init method works from a vector of components.  It will take
  /// ownership of the pointers and will resize the vector to zero to avoid a
  /// chance of the caller deallocating them (but the vector itself is not
  /// deleted).
  void Init(std::vector<Component*> *components);

  /// Appends this component to the components already in the neural net.
  /// Takes ownership of the pointer.
  void Append(Component *new_component);
  
  virtual ~Nnet() { Destroy(); }

  std::string Info() const; // some human-readable summary info.

  void Destroy();
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void SetZero(bool treat_as_gradient); // Sets all parameters to zero and if
  // treat_as_gradient == true, also tells components to "think of themselves as
  // gradients" (affects some of the update code).  Also zeroes stats stored
  // with things of type NonlinearComponent.


  /// This function is used when doing transfer learning to a new system.  It
  /// resizes the final affine and softmax components.  If your system has a
  /// SumGroupComponent before the final softmax, it will be discarded.
  void ResizeOutputLayer(int32 new_num_pdfs);
  

  /// Scale all the learning rates in the neural net by this factor.
  void ScaleLearningRates(BaseFloat factor);

  /// Scale all the learning rates in the neural net by the factors indexed
  /// by the type of component.
  void ScaleLearningRates(std::map<std::string, BaseFloat> scale_factors);

  /// Set all the learning rates in the neural net to this value.
  void SetLearningRates(BaseFloat learning_rates);

  /// Set all the learning rates in the neural net to these values
  /// (one for each updatable layer).
  void SetLearningRates(const VectorBase<BaseFloat> &learning_rates);

  /// Get all the learning rates in the neural net (the output
  /// must have dim equal to NumUpdatableComponents()).
  void GetLearningRates(VectorBase<BaseFloat> *learning_rates) const;
  
  // This sets *dot_prod to the dot prod of *this . validation_gradient,
  // separately for each updatable component.  The vector must have size equal
  // to this->NumUpdatableComponents().  Warning: previously it had to have size
  // equal to this->NumComponents()).  This is used in updating learning rates
  // and shrinkage rates.
  void ComponentDotProducts(
      const Nnet &other,
      VectorBase<BaseFloat> *dot_prod) const;

  void Check() const; // Consistency check.


  void ResetGenerators(); // resets random-number generators for all
  // random components.  You must also set sRand() for this to be
  // effective.

  // The following three functions are used for vectorizing
  // the parameters-- used, for example, in L-BFGS.
  virtual int32 GetParameterDim() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  
  friend class NnetUpdater;
  friend class DecodableNnet;
 private:
  std::vector<Component*> components_;
};


/// This function generates a random neural net, for testing purposes.
/// It will contain a random number of SigmoidComponent, AffineComponent
/// and SpliceComponent, followed by a final AffineComponent and
/// SoftmaxComponent.  The parameters will all be randomly initialized.
Nnet *GenRandomNnet(int32 input_dim,
                    int32 output_dim);



} // namespace nnet2
} // namespace kaldi

#endif
