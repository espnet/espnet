// nnet3/nnet-component-itf.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen
//                2015  Xiaohui Zhang

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

#ifndef KALDI_NNET3_NNET_COMPONENT_ITF_H_
#define KALDI_NNET3_NNET_COMPONENT_ITF_H_

#include <iostream>
#include "nnet3/nnet-common.h"
#include "nnet3/nnet-parse.h"
#include "base/kaldi-error.h"

namespace kaldi {
namespace nnet3 {

// enum used to store various binary component properties.
// We give it a name ComponentProperties, but don't use this
// type for the bitmasks: instead use int32 for this type, e.g.
// int32 properties = kSimpleComponent|kBackpropNeedsOutput.
enum ComponentProperties {
  kSimpleComponent = 0x001,  // true if number of rows of input equals number of rows
                             // of output and this component doesn't care about the indexes
                             // (i.e. it maps each row of input to each row of output without
                             // regard to the index values).  Will normally be true.
  kUpdatableComponent = 0x002,  // true if the component has parameters that can
                                // be updated.  Components that return this flag
                                // must be dynamic_castable to type
                                // UpdatableComponent (but components of type
                                // UpdatableComponent do not have to return this
                                // flag, e.g.  if this instance is not really
                                // updatable).
  kPropagateInPlace = 0x004,  // true if we can do the propagate operation in-place
                              // (input and output matrices are the same).
                              // Note: if doing backprop, you'd also need to check
                              // that the kBackpropNeedsInput property is not true.
  kPropagateAdds = 0x008,  // true if the Propagate function adds to, rather
                           // than setting, its output, for non-in-place
                           // propagation.  The Component chooses whether to add
                           // or set, and the calling code has to accommodate
                           // it.
  kReordersIndexes = 0x010,  // true if the ReorderIndexes function might reorder
                             // the indexes (otherwise we can skip calling it).
                             // Must not be set for simple components.
  kBackpropAdds = 0x020,   // true if the Backprop function adds to, rather than
                           // setting, the "in_deriv" output for non-in-place
                           // backprop.  The Component chooses whether to add or
                           // set, and the calling code has to accommodate it.
  kBackpropNeedsInput = 0x040,  // true if backprop operation needs access to
                                // forward-pass input.
  kBackpropNeedsOutput = 0x080,  // true if backprop operation needs access to
                                 // forward-pass output (e.g. true for Sigmoid).
  kBackpropInPlace = 0x100,   // true if we can do the backprop operation in-place
                             // (input and output matrices may be the same).
  kStoresStats = 0x200,      // true if the StoreStats operation stores
                             // statistics e.g. on average node activations and
                             // derivatives of the nonlinearity, (as it does for
                             // Tanh, Sigmoid, ReLU and Softmax).
  kInputContiguous = 0x400,  // true if the component requires its input data (and
                              // input derivatives) to have Stride()== NumCols().
  kOutputContiguous = 0x800,  // true if the component requires its input data (and
                               // output derivatives) to have Stride()== NumCols().
  kUsesMemo = 0x1000,  // true if the component returns a void* pointer from its
                       // Propagate() function that needs to be passed into the
                       // corresponding Backprop function.
  kRandomComponent = 0x2000   // true if the component has some kind of
                              // randomness, like DropoutComponent (these should
                              // inherit from class RandomComponent.
};


// This is a base class for a helper-class of class Component, which is used to
// store any pre-computed indexes it needs for its forward and backward
// computations.  For components which are not "Simple" components (i.e. the
// kSimpleComponent property is false), and which may therefore "care" about
// which index the input and output matrix's rows represent (i.e. about
// which "struct Index" each row corresponds to), their CreateIndexes() function
// will be called prior to Propagate() and Backprop(), to create an object which
// must be a child class of class ComponentPrecomputedIndexes, where they
// can store any indexes that they need.
class ComponentPrecomputedIndexes {
 public:
  virtual ComponentPrecomputedIndexes *Copy() const = 0;
  virtual void Write(std::ostream &os, bool binary) const = 0;
  virtual void Read(std::istream &os, bool binary) = 0;
  virtual std::string Type() const = 0;
  static ComponentPrecomputedIndexes* ReadNew(std::istream &is, bool binary);
  // cpi stands for component_precomputed_indexes
  static ComponentPrecomputedIndexes* NewComponentPrecomputedIndexesOfType(
                                           const std::string &cpi_type);
  virtual ~ComponentPrecomputedIndexes() { }
};


class IndexSet;  // Forward declaration; declared in nnet-computation-graph.h.

/// Abstract base-class for neural-net components.
class Component {
 public:
  /// \brief Propagate function.
  ///   \param [in] indexes  A pointer to some information output by this class's
  ///      PrecomputeIndexes function (will be NULL for simple components,
  ///      i.e. those that don't do things like splicing).
  ///   \param [in] in   The input to this component.  Num-columns == InputDim().
  ///   \param [out] out  The output of this component.  Num-columns == OutputDim().
  ///      Note: output of this component will be added to the initial value of
  ///      "out" if Properties()&kPropagateAdds != 0; otherwise the output will
  ///      be set and the initial value ignored.  Each Component chooses whether
  ///      it is more convenient implementation-wise to add or set, and the
  ///      calling code has to deal with it.
  ///   \return  Normally returns NULL, but may return a non-NULL value for
  ///      components which have the flag kUsesMemo set.  This value will
  ///      be passed into the corresponding Backprop routine.
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const = 0;

  /// \brief Backprop function; depending on which of the arguments 'to_update'
  ///     and 'in_deriv' are non-NULL, this can compute input-data derivatives
  ///     and/or perform model update.
  ///
  ///   \param [in] debug_info  The component name, to be printed out in any
  ///       warning messages.
  ///   \param [in] indexes     A pointer to some information output by this
  ///      class's PrecomputeIndexes function (will be NULL for simple
  ///      components, i.e. those that don't do things like splicing).
  ///   \param [in] in_value    The matrix that was given as input to the
  ///      Propagate function.  Will be ignored (and may be empty) if
  ///      Properties()&kBackpropNeedsInput == 0.
  ///   \param [in] out_value   The matrix that was output from the Propagate
  ///      function.  Will be ignored (and may be empty) if
  ///      Properties()&kBackpropNeedsOutput == 0
  ///   \param [in] out_deriv  The derivative at the output of this component.
  ///   \param [in] memo       This will normally be NULL, but for component
  ///       types that set the flag kUsesMemo, this will be the return value
  ///       of the Propagate() function that corresponds to this Backprop()
  ///       function.  Ownership of any pointers is not transferred to the
  ///       Backprop function; DeleteMemo() will be called to delete it.
  ///   \param [out] to_update  If model update is desired, the Component
  ///       to be updated, else NULL.  Does not have to be identical to this.
  ///       If supplied, you can assume that
  ///       to_update->Properties() & kUpdatableComponent is nonzero.
  ///   \param [out] in_deriv   The derivative at the input of this component,
  ///       if needed (else NULL).   If  Properties()&kBackpropInPlace, may be
  ///       the same matrix as out_deriv.  If Properties()&kBackpropAdds, this
  ///       is added to by the Backprop routine, else it is set.  The component
  ///       code chooses which mode to work in, based on convenience.
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update, // may be NULL; may be identical
                                              // to "this" or different.
                        CuMatrixBase<BaseFloat> *in_deriv) const = 0;

  /// \brief This function may store stats on average activation values, and for
  ///        some component types, the average value of the derivative of the
  ///        nonlinearity.  It only does something for those components that
  ///        have nonzero Properties()&kStoresStats.
  ///
  /// \param [in] in_value  The input to the Propagate() function.  Note: if
  ///        the component sets the flag kPropagateInPlace, this should not
  ///        be used; the empty matrix will be provided here if in-place
  ///        propagation was used.
  /// \param [in] out_value  The output of the Propagate() function.
  /// \param [in] memo  The 'memo' returned by the Propagate() function; this
  ///        will usually be NULL.
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo) { }

  /// \brief Components that provide an implementation of StoreStats should also
  ///        provide an implementation of ZeroStats(), to set those stats to
  ///        zero.  Other components that store other types of statistics
  ///        (e.g. regarding gradient clipping) should implement ZeroStats()
  ///        also.
  virtual void ZeroStats() { }



  /// \brief  This function only does something interesting for non-simple Components.
  ///   For a given index at the output of the component, tells us what indexes
  ///   are required at its input (note: "required" encompasses also optionally-required
  ///   things; it will enumerate all things that we'd like to have).  See also
  ///   IsComputable().
  /// \param [in] misc_info  This argument is supplied to handle things that the
  ///       framework can't very easily supply: information like which time
  ///       indexes are needed for AggregateComponent, which time-indexes are
  ///       available at the input of a recurrent network, and so on.  We will
  ///       add members to misc_info as needed.
  /// \param [in] output_index  The Index at the output of the component, for
  ///       which we are requesting the list of indexes at the component's input.
  /// \param [out] desired_indexes A list of indexes that are desired at the
  ///       input.  are to be written to here.  By "desired" we mean required or
  ///       optionally-required.
  ///
  /// The default implementation of this function is suitable for any
  /// SimpleComponent; it just copies the output_index to a single identical
  /// element in input_indexes.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  /// \brief This function only does something interesting for non-simple
  ///    Components, and it exists to make it possible to manage
  ///    optionally-required inputs.  It tells the user whether a given output
  ///    index is computable from a given set of input indexes, and if so,
  ///    says which input indexes will be used in the computation.
  ///
  ///    Implementations of this function are required to have the property that
  ///    adding an element to "input_index_set" can only ever change IsComputable
  ///    from false to true, never vice versa.
  ///
  ///    @param [in] misc_info  Some information specific to the computation, such as
  ///              minimum and maximum times for certain components to do adaptation on;
  ///              it's a place to put things that don't easily fit in the framework.
  ///    @param [in] output_index  The index that is to be computed at the output
  ///              of this Component.
  ///    @param [in] input_index_set  The set of indexes that is available at the
  ///              input of this Component.
  ///    @param [out] used_inputs If this is non-NULL and the output is
  ///       computable this will be set to the list of input indexes that will
  ///       actually be used in the computation.
  ///    @return Returns true iff this output is computable from the provided
  ///          inputs.
  ///
  ///   The default implementation of this function is suitable for any
  ///   SimpleComponent: it just returns true if output_index is in
  ///   input_index_set, and if so sets used_inputs to vector containing that
  ///   one Index.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  /// \brief This function only does something interesting for non-simple
  ///  Components.  It provides an opportunity for a Component to reorder the or
  ///  pad the indexes at its input and output.  This might be useful, for
  ///  instance, if a component requires a particular ordering of the indexes
  ///  that doesn't correspond to their natural ordering.  Components that might
  ///  modify the indexes are required to return the kReordersIndexes flag in
  ///  their Properties().
  ///     The ReorderIndexes() function is now allowed to insert blanks
  ///  into the indexes.  The 'blanks' must be of the form (n,kNoTime,x),
  ///  where the marker kNoTime (a very negative number) is there where
  ///  the 't' indexes normally live.  The reason we don't just have, say,
  ///  (-1,-1,-1), relates to the need to preserve a regular pattern over
  ///  the 'n' indexes so that 'shortcut compilation' (c.f. ExpandComputation())
  ///  can work correctly
  ///
  ///
  ///  \param [in,out]  Indexes at the input of the Component.
  ///  \param [in,out]  Indexes at the output of the Component
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const {}



  /// \brief This function must return NULL for simple Components.  Returns a
  ///     pointer to a class that may contain some precomputed
  ///     component-specific and computation-specific indexes to be in used in
  ///     the Propagate and Backprop functions.
  ///
  /// \param [in] misc_info  This argument is supplied to handle things that the
  ///       framework can't very easily supply: information like which time
  ///       indexes are needed for AggregateComponent, which time-indexes are
  ///       available at the input of a recurrent network, and so on.  misc_info
  ///       may not even ever be used here.  We will add members to misc_info as
  ///       needed.
  /// \param [in] input_indexes  A vector of indexes that explains
  ///       what time-indexes (and other indexes) each row of the
  ///       in/in_value/in_deriv matrices given to Propagate and Backprop will
  ///       mean.
  /// \param [in] output_indexes  A vector of indexes that explains
  ///       what time-indexes (and other indexes) each row of the
  ///       out/out_value/out_deriv matrices given to Propagate and Backprop will
  ///       mean.
  /// \param [in] need_backprop  True if we might need to do backprop
  ///       with this component, so that if any different indexes are needed
  ///       for backprop then those should be computed too.
  /// \return  Returns a child-class of class ComponentPrecomputedIndexes, or
  ///       NULL if this component for does not need to precompute any indexes
  ///       (e.g. if it is a simple component and does not care about indexes).
  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const { return NULL;  }


  /// \brief Returns a string such as "SigmoidComponent", describing
  ///        the type of the object.
  virtual std::string Type() const = 0;

  /// \brief  Initialize, from a ConfigLine object.
  /// \param [in] cfl  A ConfigLine containing any parameters that
  ///            are needed for initialization. For example:
  ///            "dim=100 param-stddev=0.1"
  virtual void InitFromConfig(ConfigLine *cfl) = 0;

  /// \brief Returns input-dimension of this component.
  virtual int32 InputDim() const = 0;

  /// \brief Returns output-dimension of this component.
  virtual int32 OutputDim() const = 0;

  /// \brief Return bitmask of the component's properties.
  ///   These properties depend only on the component's type.
  ///   See enum ComponentProperties.
  virtual int32 Properties() const = 0;

  /// \brief Read component from stream (works out its type).  Dies on error.
  static Component* ReadNew(std::istream &is, bool binary);

  /// \brief Copies component (deep copy).
  virtual Component* Copy() const = 0;

  /// \brief Returns a new Component of the given type e.g. "SoftmaxComponent",
  ///   or NULL if no such component type exists.
  static Component *NewComponentOfType(const std::string &type);

  /// \brief Read function (used after we know the type of the Component);
  ///   accepts input that is missing the token that describes the component
  ///   type, in case it has already been consumed.
  virtual void Read(std::istream &is, bool binary) = 0;

  /// \brief Write component to stream
  virtual void Write(std::ostream &os, bool binary) const = 0;

  /// \brief Returns some text-form information about this component, for diagnostics.
  ///     Starts with the type of the component.  E.g. "SigmoidComponent dim=900",
  ///     although most components will have much more info.
  virtual std::string Info() const;

  /// This virtual function when called on
  ///    -- an UpdatableComponent scales the parameters
  ///      by "scale" when called by an UpdatableComponent.
  ///    -- a Nonlinear component (or another component that
  ///      stores stats, like BatchNormComponent)-- it relates
  ///      to scaling activation stats, not parameters.
  /// Otherwise it will normally do nothing.
  virtual void Scale(BaseFloat scale) {};

  /// This virtual function when called by
  ///    -- an UpdatableComponent adds the parameters of
  ///      another updatable component, times some constant, to the current
  ///      parameters.
  ///    -- a NonlinearComponent (or another component that stores
  ///       stats, like BatchNormComponent)-- it relates to adding
  ///       stats.
  /// Otherwise it will normally do nothing.
  virtual void Add(BaseFloat alpha, const Component &other) {};

  /// This virtual function only needs to be overwritten by Components that
  /// return a non-NULL memo from their Propagate() function.  It's called by
  /// NnetComputer in cases where Propagate returns a memo but there will be no
  /// backprop to consume it.
  virtual void DeleteMemo(void *memo) const { KALDI_ASSERT(memo == NULL); }


  Component() { }

  virtual ~Component() { }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(Component);
};


class RandomComponent: public Component {
 public:
  // This function is required in testing code and in other places we need
  // consistency in the random number generation (e.g. when optimizing
  // validation-set performance), but check where else we call srand().  You'll
  // need to call srand prior to making this call.
  void ResetGenerator() { random_generator_.SeedGpu(); }

  // Call this with 'true' to set 'test mode' where the behavior is different
  // from normal mode.
  void SetTestMode(bool test_mode) { test_mode_ = test_mode; }

  RandomComponent(): test_mode_(false) { }

  RandomComponent(const RandomComponent &other):
      test_mode_(other.test_mode_) {}
 protected:
  CuRand<BaseFloat> random_generator_;

  // This is true if we want a different behavior for inference from  that for
  // training.
  bool test_mode_;
};

/**
   Class UpdatableComponent is a Component which has trainable parameters; it
   extends the interface of Component.  This is a base-class for Components with
   parameters.  See comment by declaration of kUpdatableComponent.
   The functions in this interface must only be called if the component returns
   the kUpdatable flag.

   Child classes support the following config-line parameters in addition
   to more specific ones:

     learning-rate         e.g. learning-rate=1.0e-05.  default=0.001
                           It's not normally necessary or desirable to set this
                           in the config line, as it typically gets set
                           in the training scripts.
     learning-rate-factor  e.g. learning-rate-factor=0.5, can be used to
                           conveniently control per-layer learning rates (it's
                           multiplied by the learning rates given to the
                           --learning-rate option to nnet3-copy or any
                           'set-learning-rate' directives to the
                           --edits-config option of nnet3-copy.  default=1.0.
     max-change            e.g. max-change=0.75.  Maximum allowed parameter change
                           for the parameters of this component, in Euclidean norm,
                           per update step.  If zero, no limit is applied at this
                           level (the global --max-param-change option will still
                           apply).  default=0.0.
 */
class UpdatableComponent: public Component {
 public:
  UpdatableComponent(const UpdatableComponent &other);

  // If these defaults are changed, the defaults in
  // InitLearningRatesFromConfig() should be changed too.
  UpdatableComponent(): learning_rate_(0.001), learning_rate_factor_(1.0),
                        l2_regularize_(0.0), is_gradient_(false),
                        max_change_(0.0) { }

  virtual ~UpdatableComponent() { }

  /// \brief Computes dot-product between parameters of two instances of a
  ///  Component.  Can be used for computing parameter-norm of an
  ///  UpdatableComponent.
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const = 0;

  /// This function is to be used in testing.  It adds unit noise times "stddev"
  /// to the parameters of the component.
  virtual void PerturbParams(BaseFloat stddev) = 0;

  /// Sets the learning rate of gradient descent- gets multiplied by
  /// learning_rate_factor_.
  virtual void SetUnderlyingLearningRate(BaseFloat lrate) {
    learning_rate_ = lrate * learning_rate_factor_;
  }

  /// Sets the learning rate directly, bypassing learning_rate_factor_.
  virtual void SetActualLearningRate(BaseFloat lrate) { learning_rate_ = lrate; }

  /// \brief Sets is_gradient_ to true and sets learning_rate_ to 1, ignoring
  /// learning_rate_factor_.
  virtual void SetAsGradient() { learning_rate_ = 1.0; is_gradient_ = true; }

  virtual BaseFloat LearningRateFactor() { return learning_rate_factor_; }

  // Sets the learning rate factors to lrate_factor.
  virtual void SetLearningRateFactor(BaseFloat lrate_factor) {
    learning_rate_factor_ = lrate_factor;
  }

  // Copies the learning-rate, learning-rate-factor, l2-regularize, is-gradient
  // and max-change values from 'other'.
  void SetUpdatableConfigs(const UpdatableComponent &other);

  /// freezes/unfreezes NaturalGradient updates, if applicable (to be overriden
  /// by components that use Natural Gradient).
  virtual void FreezeNaturalGradient(bool freeze) { }

  /// Gets the learning rate to be used in gradient descent.
  BaseFloat LearningRate() const { return learning_rate_; }

  /// Returns the per-component max-change value, which is interpreted as the
  /// maximum change (in l2 norm) in parameters that is allowed per minibatch
  /// for this component.  The components themselves do not enforce the
  /// per-component max-change; it's enforced in class NnetTrainer by querying
  /// the max-changes for each component.  See
  /// NnetTrainer::UpdateParamsWithMaxChange() in nnet-utils.h.
  BaseFloat MaxChange() const { return max_change_; }

  void SetMaxChange(BaseFloat max_change) { max_change_ = max_change; }

  /// Returns the l2 regularization constant, which may be set in any updatable
  /// component (usually from the config file).  This value is not interrogated
  /// in the component-level code.  Instead it is read by the function
  /// ApplyL2Regularization(), declared in nnet-utils.h, which is used as part
  /// of the training workflow.
  BaseFloat L2Regularization() const { return l2_regularize_; }

  void SetL2Regularization(BaseFloat a) { l2_regularize_ = a; }

  virtual std::string Info() const;

  /// The following new virtual function returns the total dimension of
  /// the parameters in this class.
  virtual int32 NumParameters() const { KALDI_ASSERT(0); return 0; }

  /// Turns the parameters into vector form.  We put the vector form on the CPU,
  /// because in the kinds of situations where we do this, we'll tend to use
  /// too much memory for the GPU.
  virtual void Vectorize(VectorBase<BaseFloat> *params) const { KALDI_ASSERT(0); }
  /// Converts the parameters from vector form.
  virtual void UnVectorize(const VectorBase<BaseFloat> &params) {
    KALDI_ASSERT(0);
  }

 protected:
  // to be called from child classes, extracts any learning rate information
  // from the config line and sets them appropriately.
  void InitLearningRatesFromConfig(ConfigLine *cfl);

  // To be used in child-class Read() functions, this function reads the opening
  // tag <ThisComponentType> and the learning-rate factor and the learning-rate.
  //
  // Its return value may not always be needed to be inspected by calling code;
  // if there was a token that it read but could not process it returns it, else
  // it returns "".
  std::string ReadUpdatableCommon(std::istream &is, bool binary);

  // To be used in child-class Write() functions, writes the opening
  // <ThisComponentType> tag and the learning-rate factor (if not 1.0) and the
  // learning rate;
  void WriteUpdatableCommon(std::ostream &is, bool binary) const;

  BaseFloat learning_rate_; ///< learning rate (typically 0.0..0.01)
  BaseFloat learning_rate_factor_; ///< learning rate factor (normally 1.0, but
                                   ///< can be set to another < value so that
                                   ///when < you call SetLearningRate(), that
                                   ///value will be scaled by this factor.
  BaseFloat l2_regularize_;  ///< L2 regularization constant.  See comment for
                             ///< the L2Regularization() for details.
  bool is_gradient_;  ///< True if this component is to be treated as a gradient rather
                      ///< than as parameters.  Its main effect is that we disable
                      ///< any natural-gradient update and just compute the standard
                      ///< gradient.
  BaseFloat max_change_; ///< configuration value for imposing max-change

 private:
  const UpdatableComponent &operator = (const UpdatableComponent &other); // Disallow.
};


/*   NonlinearComponent is a base-class for things like sigmoid, softmax and
     ReLU: nonlinearities that don't change the dimension.  This base-class
     takes care of storing statistics on the average activations and derivatives
     encountered during training, and model initialization and I/O.

     Supported parameters on the config line:

       dim           Dimension of the input and output of the component.
                     (Caution: for NormalizeComponent, there is a member
                     "add-log-stddev" which if true,  will increase the output
                     dim by one, so it will be "dim" plus one.

       self-repair-scale=0.0   A scale for the self-repair mechanism (which nudges
                     the activation values towards the 'good' regions when a particular
                     dimension of the activations seem to be oversaturated or otherwise
                     unbalanced.  This is typically set from the script level to values
                     like 1.0e-04 to 1.0e-05.

       self-repair-lower-threshold=-1000  A lower threshold for the self-repair mechanism;
                     it will be interpreted in a component-specific way, typically a lower
                     limit on the average derivative or activation below which the
                     self-repair mechanism is activated.  -1000 is a special value which
                     will cause a component-specific default to be used.

       self-repair-upper-threshold=-1000  An upper threshold for the self-repair mechanism;
                     it will be interpreted in a component-specific way, typically an upper
                     limit on the average derivative or activation above which the
                     self-repair mechanism is activated.  -1000 is a special value which
                     will cause a component-specific default to be used.

       block-dim     Defaults to dim, but may be any divisor of dim.  It affects the
                     self-repair, which will be done while treating the input/output as
                     repeating blocks of size 'block-dim' (e.g. blocks of filters).  It allows
                     us to do self-repair on the filter level in CNNs.
                     Currently this only makes a difference for RectifiedLinearComponent.
*/
class NonlinearComponent: public Component {
 public:

  NonlinearComponent();
  explicit NonlinearComponent(const NonlinearComponent &other);

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }

  // We implement InitFromConfig at this level and this version is sufficient
  // for most of the child classes.  Note: it's overridden by class
  // NormalizeComponent.
  virtual void InitFromConfig(ConfigLine *cfl);

  /// We implement Read at this level as it just needs the Type().
  virtual void Read(std::istream &is, bool binary);

  virtual void ZeroStats();

  virtual std::string Info() const;

  /// Write component to stream.
  virtual void Write(std::ostream &os, bool binary) const;

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);

  // The following functions are unique to NonlinearComponent.
  // They mostly relate to diagnostics.
  const CuVector<double> &ValueSum() const { return value_sum_; }
  const CuVector<double> &DerivSum() const { return deriv_sum_; }

  double Count() const { return count_; }

 protected:
  enum { kUnsetThreshold = -1000 };

  friend class SigmoidComponent;
  friend class TanhComponent;
  friend class SoftmaxComponent;
  friend class LogSoftmaxComponent;
  friend class RectifiedLinearComponent;

  // This function updates the stats "value_sum_", "deriv_sum_", and
  // count_. (If deriv == NULL, it won't update "deriv_sum_").
  // It will be called from the Backprop function of child classes.
  void StoreStatsInternal(const CuMatrixBase<BaseFloat> &out_value,
                          const CuMatrixBase<BaseFloat> *deriv = NULL);

  // This function may be called from child class members during backprop.  It
  // stores the 'oderiv_sumsq_' stats.
  void StoreBackpropStats(const CuMatrixBase<BaseFloat> &out_deriv);


  const NonlinearComponent &operator = (const NonlinearComponent &other); // Disallow.

  // dim_ is the input dimension (and almost always the output dimension) of the
  // component.
  int32 dim_;
  // block_dim_ will normally be the same as dim_, but it may be any nonzero
  // divisor of dim_; if so, each vector is treated as a number of blocks
  // appended together, and this affects the stats accumulation and self-repair.
  // Currently this is only supported for RectifiedLinearComponent.
  int32 block_dim_;
  CuVector<double> value_sum_; // stats at the output.
  CuVector<double> deriv_sum_; // stats of the derivative of the nonlinearity
                               // (only applicable to element-by-element
                               // nonlinearities, not Softmax.
  // Count corresponding to the stats in 'value_sum_' and 'deriv_sum_'
  double count_;

  CuVector<double> oderiv_sumsq_;  // Sum-square of the derivative of the
                                   // objective function, that we're propagating
                                   // back.  Accumulated during the backprop;
                                   // used for diagnostics.
  // Count corresponding to the stats in 'oderiv_sumsq_'.
  double oderiv_count_;

  // some stats for self-repairing nonlinearities.
  double num_dims_self_repaired_;
  double num_dims_processed_;

  // some configuration values relating to self-repairing nonlinearities.
  BaseFloat self_repair_lower_threshold_;
  BaseFloat self_repair_upper_threshold_;
  BaseFloat self_repair_scale_;
};

} // namespace nnet3
} // namespace kaldi


#endif
