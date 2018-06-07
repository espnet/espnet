// nnet3/nnet-discriminative-example.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//           2014-2015  Vimal Manohar

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

#ifndef KALDI_NNET3_NNET_DISCRIMINATIVE_EXAMPLE_H_
#define KALDI_NNET3_NNET_DISCRIMINATIVE_EXAMPLE_H_

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation.h"
#include "util/table-types.h"
#include "nnet3/discriminative-supervision.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"

namespace kaldi {
namespace nnet3 {

// Glossary: mmi = Maximum Mutual Information,
//          mpfe = Minimum Phone Frame Error
//          smbr = State-level Minimum Bayes Risk

// This file relates to the creation of examples for discriminative training

struct NnetDiscriminativeSupervision {
  // the name of the output in the neural net; in simple setups it
  // will just be "output".
  std::string name;

  // The indexes that the output corresponds to.  The size of this vector will
  // be equal to supervision.num_sequences * supervision.frames_per_sequence.
  // Be careful about the order of these indexes-- it is a little confusing.
  // The indexes in the 'index' vector are ordered as: (frame 0 of each sequence);
  // (frame 1 of each sequence); and so on.  But in the 'supervision' object,
  // the lattice contains (sequence 0; sequence 1; ...).  So reordering is needed.
  // This is done to make the code similar that for the 'chain' model.
  std::vector<Index> indexes;

  // The supervision object, containing the numerator and denominator
  // lattices.
  discriminative::DiscriminativeSupervision supervision;

  // This is a vector of per-frame weights, required to be between 0 and 1,
  // that is applied to the derivative during training (but not during model
  // combination, where the derivatives need to agree with the computed objf
  // values for the optimization code to work).  The reason for this is to more
  // exactly handle edge effects and to ensure that no frames are
  // 'double-counted'.  The order of this vector corresponds to the order of
  // the 'indexes' (i.e. all the first frames, then all the second frames,
  // etc.)
  // If this vector is empty it means we're not applying per-frame weights,
  // so it's equivalent to a vector of all ones.  This vector is written
  // to disk compactly as unsigned char.
  Vector<BaseFloat> deriv_weights;

  // Use default assignment operator
  NnetDiscriminativeSupervision() { }

  // Initialize the object from an object of type discriminative::Supervision,
  // and some extra information.
  // Note: you probably want to set 'name' to "output".
  // 'first_frame' will often be zero but you can choose (just make it
  // consistent with how you numbered your inputs), and 'frame_skip' would be 1
  // in a vanilla setup, but 3 in the case of 'chain' models
  NnetDiscriminativeSupervision(const std::string &name,
                                const discriminative::DiscriminativeSupervision &supervision,
                                const VectorBase<BaseFloat> &deriv_weights,
                                int32 first_frame,
                                int32 frame_skip);

  NnetDiscriminativeSupervision(const NnetDiscriminativeSupervision &other);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void Swap(NnetDiscriminativeSupervision *other);

  void CheckDim() const;

  bool operator == (const NnetDiscriminativeSupervision &other) const;
};

/// NnetDiscriminativeExample is like NnetExample, but specialized for
/// sequence training.
struct NnetDiscriminativeExample {

  /// 'inputs' contains the input to the network-- normally just it has just one
  /// element called "input", but there may be others (e.g. one called
  /// "ivector")...  this depends on the setup.
  std::vector<NnetIo> inputs;

  /// 'outputs' contains the sequence output supervision.  There will normally
  /// be just one member with name == "output".
  std::vector<NnetDiscriminativeSupervision> outputs;

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void Swap(NnetDiscriminativeExample *other);

  // Compresses the input features (if not compressed)
  void Compress();

  NnetDiscriminativeExample() { }

  NnetDiscriminativeExample(const NnetDiscriminativeExample &other);

  bool operator == (const NnetDiscriminativeExample &other) const {
    return inputs == other.inputs && outputs == other.outputs;
  }
};


/// This hashing object hashes just the structural aspects of the NnetExample
/// without looking at the value of the features.  It will be used in combining
/// egs into batches of all similar structure.
struct NnetDiscriminativeExampleStructureHasher {
  size_t operator () (const NnetDiscriminativeExample &eg) const noexcept ;
  // We also provide a version of this that works from pointers.
  size_t operator () (const NnetDiscriminativeExample *eg) const noexcept {
    return (*this)(*eg);
  }
};


/// This comparator object compares just the structural aspects of the
/// NnetDiscriminativeExample without looking at the value of the features.
struct NnetDiscriminativeExampleStructureCompare {
  bool operator () (const NnetDiscriminativeExample &a,
                    const NnetDiscriminativeExample &b) const;
  // We also provide a version of this that works from pointers.
  bool operator () (const NnetDiscriminativeExample *a,
                    const NnetDiscriminativeExample *b) const {
    return (*this)(*a, *b);
  }
};


/**
  Appends the given vector of examples (which must be non-empty) into
  a single output example.
  Intended to be used when forming minibatches for neural net training. If
  'compress' it compresses the output features (recommended to save disk
  space).

  Note: the input is left as it was at the start, but it is temporarily
  changed inside the function; this is a trick to allow us to use the
  MergeExamples() routine while avoiding having to rewrite code.
*/
void MergeDiscriminativeExamples(
    std::vector<NnetDiscriminativeExample> *input,
    bool compress,
    NnetDiscriminativeExample *output);

// called from MergeDiscriminativeExamples, this function merges the Supervision
// objects into one.  Requires (and checks) that they all have the same name.
void MergeSupervision(
    const std::vector<const NnetDiscriminativeSupervision*> &inputs,
    NnetDiscriminativeSupervision *output);


/** Shifts the time-index t of everything in the input of "eg" by adding
    "t_offset" to all "t" values-- but excluding those with names listed in
    "exclude_names", e.g.  "ivector".  This might be useful if you are doing
    subsampling of frames at the output, because shifted examples won't be quite
    equivalent to their non-shifted counterparts.  "exclude_names" is a vector
    of names of nnet inputs that we avoid shifting the "t" values of-- normally
    it will contain just the single string "ivector" because we always leave t=0
    for any ivector.

    Note: input features will be shifted by 'frame_shift', and indexes in the
    supervision in (eg->output) will be shifted by 'frame_shift' rounded to the
    closest multiple of the frame subsampling factor (e.g. 3).  The frame
    subsampling factor is worked out from the time spacing between the indexes
    in the output.  */
void ShiftDiscriminativeExampleTimes(int32 frame_shift,
                                    const std::vector<std::string> &exclude_names,
                                    NnetDiscriminativeExample *eg);

/**  This function takes a NnetDiscriminativeExample and produces a
     ComputationRequest.
     Assumes you don't want the derivatives w.r.t. the inputs; if you do, you
     can create the ComputationRequest manually.  Assumes that if
     need_model_derivative is true, you will be supplying derivatives w.r.t. all
     outputs.
*/
void GetDiscriminativeComputationRequest(const Nnet &nnet,
                                         const NnetDiscriminativeExample &eg,
                                         bool need_model_derivative,
                                         bool store_component_stats,
                                         bool use_xent_regularization,
                                         bool use_xent_derivative,
                                         ComputationRequest *computation_request);

typedef TableWriter<KaldiObjectHolder<NnetDiscriminativeExample > > NnetDiscriminativeExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetDiscriminativeExample > > SequentialNnetDiscriminativeExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetDiscriminativeExample > > RandomAccessNnetDiscriminativeExampleReader;


/// This function returns the 'size' of a discriminative example as defined for
/// purposes of merging egs, which is defined as the largest number of Indexes
/// in any of the inputs or outputs of the example.
int32 GetDiscriminativeNnetExampleSize(const NnetDiscriminativeExample &a);


/// This class is responsible for arranging examples in groups that have the
/// same strucure (i.e. the same input and output indexes), and outputting them
/// in suitable minibatches as defined by ExampleMergingConfig.
class DiscriminativeExampleMerger {
 public:
  DiscriminativeExampleMerger(const ExampleMergingConfig &config,
                              NnetDiscriminativeExampleWriter *writer);

  // This function accepts an example, and if possible, writes a merged example
  // out.  The ownership of the pointer 'a' is transferred to this class when
  // you call this function.
  void AcceptExample(NnetDiscriminativeExample *a);

  // This function announces to the class that the input has finished, so it
  // should flush out any smaller-sized minibatches, as dictated by the config.
  // This will be called in the destructor, but you can call it explicitly when
  // all the input is done if you want to; it won't repeat anything if called
  // twice.  It also prints the stats.
  void Finish();

  // returns a suitable exit status for a program.
  int32 ExitStatus() { Finish(); return (num_egs_written_ > 0 ? 0 : 1); }

  ~DiscriminativeExampleMerger() { Finish(); };
 private:
  // called by Finish() and AcceptExample().  Merges, updates the stats, and
  // writes.  The 'egs' is non-const only because the egs are temporarily
  // changed inside MergeDiscriminativeEgs.  The pointer 'egs' is still owned
  // by the caller.
  void WriteMinibatch(std::vector<NnetDiscriminativeExample> *egs);

  bool finished_;
  int32 num_egs_written_;
  const ExampleMergingConfig &config_;
  NnetDiscriminativeExampleWriter *writer_;
  ExampleMergingStats stats_;

  // Note: the "key" into the egs is the first element of the vector.
  typedef unordered_map<NnetDiscriminativeExample*,
                        std::vector<NnetDiscriminativeExample*>,
                        NnetDiscriminativeExampleStructureHasher,
                        NnetDiscriminativeExampleStructureCompare> MapType;
   MapType eg_to_egs_;
};


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_DISCRIMINATIVE_EXAMPLE_H_
