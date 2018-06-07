// nnet2/nnet-nnet.cc

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

#include <algorithm>
#include <set>
#include <string>
#include "nnet2/nnet-nnet.h"
#include "util/stl-utils.h"

namespace kaldi {
namespace nnet2 {


int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::InputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.front()->InputDim();
}


int32 Nnet::LeftContext() const {
  KALDI_ASSERT(!components_.empty());
  int32 ans = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    ans += components_[i]->Context().front();
  }
  return -1*ans;
  // nnet-components return left context as a non-positive integer
  // however the nnet-update, nnet-compute expect a
  // non-negative left context. In addition, the NnetExample also stores data
  // left context as positive integer. To be compatible with these other classes
  // Nnet::LeftContext() returns a non-negative left context.
}

int32 Nnet::RightContext() const {
  KALDI_ASSERT(!components_.empty());
  int32 ans = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    ans += components_[i]->Context().back();
  }
  return ans;
}

void Nnet::ComputeChunkInfo(int32 input_chunk_size,
                            int32 num_chunks,
                            std::vector<ChunkInfo> *chunk_info_out) const {
  // First compute the output-chunk indices for the last component in the
  // network. we assume that the numbering of the input starts from zero.
  int32 output_chunk_size = input_chunk_size - LeftContext() - RightContext();
  KALDI_ASSERT(output_chunk_size > 0);
  std::vector<int32> current_output_inds;
  for (int32 i = 0; i < output_chunk_size; i++)
    current_output_inds.push_back(i + LeftContext());

  (*chunk_info_out).resize(NumComponents() + 1);

  // indexes for last component is empty, since the last component's chunk is
  // always contiguous
  // component's output is always contiguous
  (*chunk_info_out)[NumComponents()] = ChunkInfo(
      GetComponent(NumComponents() - 1).OutputDim(),
      num_chunks, current_output_inds.front(),
      current_output_inds.back());

  std::vector<int32> current_input_inds;
  for (int32 i = NumComponents() - 1; i >= 0; i--) {
    std::vector<int32> current_context = GetComponent(i).Context();
    std::set<int32> current_input_ind_set;
    for (size_t j = 0; j < current_context.size(); j++)
      for (size_t k = 0; k < current_output_inds.size(); k++)
        current_input_ind_set.insert(current_context[j] +
                                     current_output_inds[k]);
    current_output_inds.resize(current_input_ind_set.size());
    std::copy(current_input_ind_set.begin(),
              current_input_ind_set.end(),
              current_output_inds.begin());

    // checking if the vector has contiguous data
    // assign indexes only if the data is not contiguous
    if (current_output_inds.size() !=
        current_output_inds.back() - current_output_inds.front() + 1) {
      (*chunk_info_out)[i] = ChunkInfo(GetComponent(i).InputDim(),
                                       num_chunks,
                                       current_output_inds);
    } else  {
      (*chunk_info_out)[i] = ChunkInfo(GetComponent(i).InputDim(),
                                       num_chunks,
                                       current_output_inds.front(),
                                       current_output_inds.back());
    }
  }

  // TODO: Make a set of components which can deal with data rearrangement.
  // Define this set in an appropriate place so that
  // users adding new components can simply update the list.
  const char *dinit[] = {"SpliceComponent", "SpliceMaxComponent"};
  std::vector< std::string > data_rearrange_components(dinit, dinit + 2);

  // Ensuring that all components until the first component capable of data
  // rearrangement (e.g. SpliceComponent|SpliceMaxComponent) operate on
  // contiguous chunks at the input
  for (size_t i = 0 ; i < NumComponents() ; i++) {
      (*chunk_info_out)[i].MakeOffsetsContiguous();
      // Check if the current component is present in the set of components
      // capable of data rearrangement.
      if (std::find(data_rearrange_components.begin(),
                    data_rearrange_components.end(),
                    components_[i]->Type())
          != data_rearrange_components.end())
          break;
  }

  // sanity testing for chunk_info_out vector
  for (size_t i = 0; i < chunk_info_out->size(); i++) {
    (*chunk_info_out)[i].Check();
    // (*chunk_info_out)[i].ToString();
  }
}

const Component& Nnet::GetComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

Component& Nnet::GetComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

void Nnet::SetZero(bool treat_as_gradient) {
  for (size_t i = 0; i < components_.size(); i++) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
    if (uc != NULL) uc->SetZero(treat_as_gradient);
    NonlinearComponent *nc = dynamic_cast<NonlinearComponent*>(components_[i]);
    if (nc != NULL) nc->Scale(0.0);
  }
}

void Nnet::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<Nnet>");
  int32 num_components = components_.size();
  WriteToken(os, binary, "<NumComponents>");
  WriteBasicType(os, binary, num_components);
  WriteToken(os, binary, "<Components>");
  for (int32 c = 0; c < num_components; c++) {
    components_[c]->Write(os, binary);
    if (!binary) os << std::endl;
  }
  WriteToken(os, binary, "</Components>");
  WriteToken(os, binary, "</Nnet>");
}

void Nnet::Read(std::istream &is, bool binary) {
  Destroy();
  ExpectToken(is, binary, "<Nnet>");
  int32 num_components;
  ExpectToken(is, binary, "<NumComponents>");
  ReadBasicType(is, binary, &num_components);
  ExpectToken(is, binary, "<Components>");
  components_.resize(num_components);
  for (int32 c = 0; c < num_components; c++)
    components_[c] = Component::ReadNew(is, binary);
  ExpectToken(is, binary, "</Components>");
  ExpectToken(is, binary, "</Nnet>");
  SetIndexes();
  Check();
}


void Nnet::ZeroStats() {
  for (size_t i = 0; i < components_.size(); i++) {
    NonlinearComponent *nonlinear_component =
        dynamic_cast<NonlinearComponent*>(components_[i]);
    if (nonlinear_component != NULL)
      nonlinear_component->Scale(0.0);  // Zero the stats this way.
  }
}
void Nnet::Destroy() {
  while (!components_.empty()) {
    delete components_.back();
    components_.pop_back();
  }
}

void Nnet::ComponentDotProducts(
    const Nnet &other,
    VectorBase<BaseFloat> *dot_prod) const {
  KALDI_ASSERT(dot_prod->Dim() == NumUpdatableComponents());
  int32 index = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    UpdatableComponent *uc1 = dynamic_cast<UpdatableComponent*>(components_[i]);
    const UpdatableComponent *uc2 = dynamic_cast<const UpdatableComponent*>(
        &(other.GetComponent(i)));
    KALDI_ASSERT((uc1 != NULL) == (uc2 != NULL));
    if (uc1 != NULL) {
      (*dot_prod)(index) = uc1->DotProduct(*uc2);
      index++;
    }
  }
  KALDI_ASSERT(index == NumUpdatableComponents());
}


Nnet::Nnet(const Nnet &other): components_(other.components_.size()) {
  for (size_t i = 0; i < other.components_.size(); i++)
    components_[i] = other.components_[i]->Copy();
  SetIndexes();
  Check();
}

Nnet::Nnet(const Nnet &other1, const Nnet &other2) {
  int32 dim1 = other1.OutputDim(), dim2 = other2.InputDim();
  if (dim1 != dim2)
    KALDI_ERR << "Concatenating neural nets: dimension mismatch "
              << dim1 << " vs. " << dim2;
  for (size_t i = 0; i < other1.components_.size(); i++)
    components_.push_back(other1.components_[i]->Copy());
  for (size_t i = 0; i < other2.components_.size(); i++)
    components_.push_back(other2.components_[i]->Copy());
  SetIndexes();
  Check();
}


Nnet &Nnet::operator = (const Nnet &other) {
  Destroy();
  components_.resize(other.components_.size());
  for (size_t i = 0; i < other.components_.size(); i++)
    components_[i] = other.components_[i]->Copy();
  SetIndexes();
  Check();
  return *this;
}

std::string Nnet::Info() const {
  std::ostringstream ostr;
  ostr << "num-components " << NumComponents() << std::endl;
  ostr << "num-updatable-components " << NumUpdatableComponents() << std::endl;
  ostr << "left-context " << LeftContext() << std::endl;
  ostr << "right-context " << RightContext() << std::endl;
  ostr << "input-dim " << InputDim() << std::endl;
  ostr << "output-dim " << OutputDim() << std::endl;
  ostr << "parameter-dim " << GetParameterDim() << std::endl;
  for (int32 i = 0; i < NumComponents(); i++)
    ostr << "component " << i << " : " << components_[i]->Info() << std::endl;
  return ostr.str();
}

void Nnet::Check() const {
  for (size_t i = 0; i + 1 < components_.size(); i++) {
    KALDI_ASSERT(components_[i] != NULL);
    int32 output_dim = components_[i]->OutputDim(),
      next_input_dim = components_[i+1]->InputDim();
    KALDI_ASSERT(output_dim == next_input_dim);
    KALDI_ASSERT(components_[i]->Index() == static_cast<int32>(i));
  }
}

void Nnet::Init(std::istream &is) {
  Destroy();
  std::string line;
  /* example config file as follows.  The things in brackets specify the context
     splicing for each layer, and after that is the info about the actual layer.
     Imagine the input dim is 13, and the speaker dim is 40, so (13 x 9) + 40 =  527.
     The config file might be as follows; the lines beginning with # are comments.

     # layer-type layer-options
     AffineLayer 0.01 0.001 527 1000 0.04356
  */
  components_.clear();
  while (getline(is, line)) {
    std::istringstream line_is(line);
    line_is >> std::ws;  // Eat up whitespace.
    if (line_is.peek() == '#' || line_is.eof()) continue;  // Comment or empty.
    Component *c = Component::NewFromString(line);
    KALDI_ASSERT(c != NULL);
    components_.push_back(c);
  }
  SetIndexes();
  Check();
}

void Nnet::Init(std::vector<Component*> *components) {
  Destroy();
  components_.swap(*components);
  SetIndexes();
  Check();
}


void Nnet::ScaleLearningRates(BaseFloat factor) {
  std::ostringstream ostr;
  for (int32 c = 0; c < NumComponents(); c++) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[c]);
    if (uc != NULL) {  // Updatable component...
      uc->SetLearningRate(uc->LearningRate() * factor);
      ostr << uc->LearningRate() << " ";
    }
  }
  KALDI_LOG << "Scaled learning rates by " << factor
            << ", new learning rates are "
            << ostr.str();
}

void Nnet::ScaleLearningRates(std::map<std::string, BaseFloat> scale_factors) {
  std::ostringstream ostr;
  for (int32 c = 0; c < NumComponents(); c++) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[c]);
    if (uc != NULL) {  // Updatable component...
      // check if scaling factor was specified for a component of this type
      std::map<std::string, BaseFloat>::const_iterator lr_iterator =
        scale_factors.find(uc->Type());
      if (lr_iterator != scale_factors.end())  {
        uc->SetLearningRate(uc->LearningRate() * lr_iterator->second);
        ostr << uc->LearningRate() << " ";
      }
    }
  }
  KALDI_LOG << "Scaled learning rates by component-type specific factor, "
            << "new learning rates are "
            << ostr.str();
}

void Nnet::SetLearningRates(BaseFloat learning_rate) {
  for (int32 c = 0; c < NumComponents(); c++) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[c]);
    if (uc != NULL) {  // Updatable component...
      uc->SetLearningRate(learning_rate);
    }
  }
  KALDI_LOG << "Set learning rates to " << learning_rate;
}

void Nnet::ResizeOutputLayer(int32 new_num_pdfs) {
  KALDI_ASSERT(new_num_pdfs > 0);
  KALDI_ASSERT(NumComponents() > 2);
  int32 nc = NumComponents();
  SumGroupComponent *sgc =
      dynamic_cast<SumGroupComponent*>(components_[nc - 1]);
  if (sgc != NULL) {
    // Remove it.  We'll resize things later.
    delete sgc;
    components_.erase(components_.begin() + nc - 1,
                      components_.begin() + nc);
    nc--;
  }
  SoftmaxComponent *sc;
  if ((sc = dynamic_cast<SoftmaxComponent*>(components_[nc - 1])) == NULL)
    KALDI_ERR << "Expected last component to be SoftmaxComponent.";

  // check if nc-1 has a FixedScaleComponent
  bool has_fixed_scale_component = false;
  int32 fixed_scale_component_index = -1;
  int32 final_affine_component_index = nc - 2;
  int32 softmax_component_index = nc - 1;
  FixedScaleComponent *fsc =
      dynamic_cast<FixedScaleComponent*>(
          components_[final_affine_component_index]);
  if (fsc != NULL)  {
    has_fixed_scale_component = true;
    fixed_scale_component_index = nc - 2;
    final_affine_component_index = nc - 3;
  }
  // note: it could be child class of AffineComponent.
  AffineComponent *ac = dynamic_cast<AffineComponent*>(
      components_[final_affine_component_index]);
  if (ac == NULL)
    KALDI_ERR << "Network doesn't have expected structure (didn't find final "
              << "AffineComponent).";
  if (has_fixed_scale_component)  {
    // collapse the fixed_scale_component with the affine_component before it
    AffineComponent *ac_new =
        dynamic_cast<AffineComponent*>(ac->CollapseWithNext(*fsc));
    KALDI_ASSERT(ac_new != NULL);
    delete fsc;
    delete ac;
    components_.erase(components_.begin() + fixed_scale_component_index,
                      components_.begin() + (fixed_scale_component_index + 1));
    components_[final_affine_component_index] = ac_new;
    ac = ac_new;
    softmax_component_index = softmax_component_index - 1;
  }
  ac->Resize(ac->InputDim(), new_num_pdfs);
  // Remove the softmax component, and replace it with a new one
  delete components_[softmax_component_index];
  components_[softmax_component_index] = new SoftmaxComponent(new_num_pdfs);
  this->SetIndexes();  // used for debugging
  this->Check();
}

int32 Nnet::NumUpdatableComponents() const {
  int32 ans = 0;
  for (int32 i = 0; i < NumComponents(); i++)
    if (dynamic_cast<const UpdatableComponent*>(&(GetComponent(i))) != NULL)
      ans++;
  return ans;
}

void Nnet::ScaleComponents(const VectorBase<BaseFloat> &scale_params) {
  KALDI_ASSERT(scale_params.Dim() == this->NumUpdatableComponents());
  int32 i = 0;
  for (int32 j = 0; j < NumComponents(); j++) {
    UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(GetComponent(j)));
    if (uc!= NULL) {
      uc->Scale(scale_params(i));
      i++;
    }
  }
  KALDI_ASSERT(i == scale_params.Dim());
}

// Scales all UpdatableComponents and all NonlinearComponents.
void Nnet::Scale(BaseFloat scale) {
  for (int32 i = 0; i < NumComponents(); i++) {
    UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(GetComponent(i)));
    if (uc != NULL) uc->Scale(scale);
    NonlinearComponent *nc =
        dynamic_cast<NonlinearComponent*>(&(GetComponent(i)));
    if (nc != NULL) nc->Scale(scale);
  }
}

void Nnet::CopyStatsFrom(const Nnet &other) {
  KALDI_ASSERT(NumComponents() == other.NumComponents());
  for (int32 i = 0; i < NumComponents(); i++) {
    NonlinearComponent *nc =
        dynamic_cast<NonlinearComponent*>(&(GetComponent(i)));
    const NonlinearComponent *nc_other =
        dynamic_cast<const NonlinearComponent*>(&(other.GetComponent(i)));
    if (nc != NULL) {
      nc->Scale(0.0);
      nc->Add(1.0, *nc_other);
    }
  }
}

void Nnet::SetLearningRates(const VectorBase<BaseFloat> &learning_rates) {
  KALDI_ASSERT(learning_rates.Dim() == this->NumUpdatableComponents());
  KALDI_ASSERT(learning_rates.Min() >= 0.0);  // we allow zero learning rate.
  int32 i = 0;
  for (int32 j = 0; j < NumComponents(); j++) {
    UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(GetComponent(j)));
    if (uc!= NULL) {
      uc->SetLearningRate(learning_rates(i));
      i++;
    }
  }
  KALDI_ASSERT(i == learning_rates.Dim());
}

void Nnet::GetLearningRates(VectorBase<BaseFloat> *learning_rates) const {
  KALDI_ASSERT(learning_rates->Dim() == this->NumUpdatableComponents());
  int32 i = 0;
  for (int32 j = 0; j < NumComponents(); j++) {
    const UpdatableComponent *uc =
        dynamic_cast<const UpdatableComponent*>(&(GetComponent(j)));
    if (uc!= NULL) {
      (*learning_rates)(i) = uc->LearningRate();
      i++;
    }
  }
  KALDI_ASSERT(i == learning_rates->Dim());
}

void Nnet::Resize(int32 new_size) {
  KALDI_ASSERT(new_size <= static_cast<int32>(components_.size()));
  for (size_t i = new_size; i < components_.size(); i++)
    delete components_[i];
  components_.resize(new_size);
}

void Nnet::RemoveDropout() {
  std::vector<Component*> components;
  int32 removed = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    if (dynamic_cast<DropoutComponent*>(components_[i]) != NULL ||
        dynamic_cast<AdditiveNoiseComponent*>(components_[i]) != NULL) {
      delete components_[i];
      removed++;
    } else {
      components.push_back(components_[i]);
    }
  }
  components_ = components;
  if (removed > 0)
    KALDI_LOG << "Removed " << removed << " dropout components.";
  SetIndexes();
  Check();
}

void Nnet::SetDropoutScale(BaseFloat scale) {
  size_t n_set = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    DropoutComponent *dc =
        dynamic_cast<DropoutComponent*>(components_[i]);
    if (dc != NULL) {
      dc->SetDropoutScale(scale);
      n_set++;
    }
  }
  KALDI_LOG << "Set dropout scale to " << scale
            << " for " << n_set << " components.";
}


void Nnet::RemovePreconditioning() {
  for (size_t i = 0; i < components_.size(); i++) {
    if (dynamic_cast<AffineComponentPreconditioned*>(components_[i]) != NULL) {
      AffineComponent *ac = new AffineComponent(
          *(dynamic_cast<AffineComponent*>(components_[i])));
      delete components_[i];
      components_[i] = ac;
    } else if (dynamic_cast<AffineComponentPreconditionedOnline*>(
        components_[i]) != NULL) {
      AffineComponent *ac = new AffineComponent(
          *(dynamic_cast<AffineComponent*>(components_[i])));
      delete components_[i];
      components_[i] = ac;
    }
  }
  SetIndexes();
  Check();
}


void Nnet::SwitchToOnlinePreconditioning(int32 rank_in, int32 rank_out,
                                         int32 update_period,
                                         BaseFloat num_samples_history,
                                         BaseFloat alpha) {
  int32 switched = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    if (dynamic_cast<AffineComponent*>(components_[i]) != NULL) {
      AffineComponentPreconditionedOnline *ac =
          new AffineComponentPreconditionedOnline(
              *(dynamic_cast<AffineComponent*>(components_[i])),
              rank_in, rank_out, update_period, num_samples_history, alpha);
      delete components_[i];
      components_[i] = ac;
      switched++;
    }
  }
  KALDI_LOG << "Switched " << switched << " components to use online "
            << "preconditioning, with (input, output) rank = "
            << rank_in << ", " << rank_out << " and num_samples_history = "
            << num_samples_history;
  SetIndexes();
  Check();
}


void Nnet::AddNnet(const VectorBase<BaseFloat> &scale_params,
                   const Nnet &other) {
  KALDI_ASSERT(scale_params.Dim() == this->NumUpdatableComponents());
  int32 i = 0;
  for (int32 j = 0; j < NumComponents(); j++) {
    UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(GetComponent(j)));
    const UpdatableComponent *uc_other =
        dynamic_cast<const UpdatableComponent*>(&(other.GetComponent(j)));
    if (uc != NULL) {
      KALDI_ASSERT(uc_other != NULL);
      BaseFloat alpha = scale_params(i);
      uc->Add(alpha, *uc_other);
      i++;
    }
  }
  KALDI_ASSERT(i == scale_params.Dim());
}

void Nnet::AddNnet(BaseFloat alpha,
                   const Nnet &other) {
  for (int32 i = 0; i < NumComponents(); i++) {
    UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(GetComponent(i)));
    const UpdatableComponent *uc_other =
        dynamic_cast<const UpdatableComponent*>(&(other.GetComponent(i)));
    if (uc != NULL) {
      KALDI_ASSERT(uc_other != NULL);
      uc->Add(alpha, *uc_other);
    }
    NonlinearComponent *nc =
        dynamic_cast<NonlinearComponent*>(&(GetComponent(i)));
    const NonlinearComponent *nc_other =
        dynamic_cast<const NonlinearComponent*>(&(other.GetComponent(i)));
    if (nc != NULL) {
      KALDI_ASSERT(nc_other != NULL);
      nc->Add(alpha, *nc_other);
    }
  }
}

void Nnet::AddNnet(BaseFloat alpha,
                   Nnet *other,
                   BaseFloat beta) {
  for (int32 i = 0; i < NumComponents(); i++) {
    UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(GetComponent(i)));
    UpdatableComponent *uc_other =
        dynamic_cast<UpdatableComponent*>(&(other->GetComponent(i)));
    if (uc != NULL) {
      KALDI_ASSERT(uc_other != NULL);
      uc->Add(alpha, *uc_other);
      uc_other->Scale(beta);
    }
    NonlinearComponent *nc =
        dynamic_cast<NonlinearComponent*>(&(GetComponent(i)));
    NonlinearComponent *nc_other =
        dynamic_cast<NonlinearComponent*>(&(other->GetComponent(i)));
    if (nc != NULL) {
      KALDI_ASSERT(nc_other != NULL);
      nc->Add(alpha, *nc_other);
      nc_other->Scale(beta);
    }
  }
}


void Nnet::Append(Component *new_component) {
  components_.push_back(new_component);
  SetIndexes();
  Check();
}

void Nnet::SetComponent(int32 c, Component *component) {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  delete components_[c];
  components_[c] = component;
  SetIndexes();
  Check();  // Check that all the dimensions still match up.
}

int32 Nnet::GetParameterDim() const {
  int32 ans = 0;
  for (int32 c = 0; c < NumComponents(); c++) {
    const UpdatableComponent *uc = dynamic_cast<const UpdatableComponent*>(
        &(GetComponent(c)));
    if (uc != NULL)
      ans += uc->GetParameterDim();
  }
  return ans;
}

void Nnet::Vectorize(VectorBase<BaseFloat> *params) const {
  int32 offset = 0;
  for (int32 c = 0; c < NumComponents(); c++) {
    const UpdatableComponent *uc = dynamic_cast<const UpdatableComponent*>(
        &(GetComponent(c)));
    if (uc != NULL) {
      int32 size = uc->GetParameterDim();
      SubVector<BaseFloat> temp(*params, offset, size);
      uc->Vectorize(&temp);
      offset += size;
    }
  }
  KALDI_ASSERT(offset == GetParameterDim());
}

void Nnet::ResetGenerators() {
  // resets random-number generators for all random
  // components.
  for (int32 c = 0; c < NumComponents(); c++) {
    RandomComponent *rc = dynamic_cast<RandomComponent*>(
        &(GetComponent(c)));
    if (rc != NULL)
      rc->ResetGenerator();
  }
}

void Nnet::UnVectorize(const VectorBase<BaseFloat> &params) {
  int32 offset = 0;
  for (int32 c = 0; c < NumComponents(); c++) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(
        &(GetComponent(c)));
    if (uc != NULL) {
      int32 size = uc->GetParameterDim();
      uc->UnVectorize(params.Range(offset, size));
      offset += size;
    }
  }
  KALDI_ASSERT(offset == GetParameterDim());
}

void Nnet::LimitRankOfLastLayer(int32 dim) {
  for (int32 i = components_.size() - 1; i >= 0; i--) {
    AffineComponent *a = NULL, *b = NULL,
        *c = dynamic_cast<AffineComponent*>(components_[i]);
    if (c != NULL) {
      c->LimitRank(dim, &a, &b);
      delete c;
      components_[i] = a;
      components_.insert(components_.begin() + i + 1, b);
      this->SetIndexes();
      this->Check();
      return;
    }
  }
  KALDI_ERR << "No affine component found in neural net.";
}

void Nnet::SetIndexes() {
  for (size_t i = 0; i < components_.size(); i++)
    components_[i]->SetIndex(i);
}

void Nnet::Collapse(bool match_updatableness) {
  int32 num_collapsed = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i + 1 < components_.size(); i++) {
      AffineComponent *a1 = dynamic_cast<AffineComponent*>(components_[i]),
          *a2 = dynamic_cast<AffineComponent*>(components_[i + 1]);
      FixedAffineComponent
          *f1 = dynamic_cast<FixedAffineComponent*>(components_[i]),
          *f2 = dynamic_cast<FixedAffineComponent*>(components_[i + 1]);
      Component *c = NULL;
      if (a1 != NULL && a2 != NULL) {
        c = a1->CollapseWithNext(*a2);
      } else if (a1 != NULL && f2 != NULL && !match_updatableness) {
        c = a1->CollapseWithNext(*f2);
      } else if (f1 != NULL && a2 != NULL && !match_updatableness) {
        c = a2->CollapseWithPrevious(*f1);
      }
      if (c != NULL) {
        delete components_[i];
        delete components_[i + 1];
        components_[i] = c;
        // This was causing valgrind errors, so doing it differently.  Either
        // a standard-library bug or I misunderstood something.
        // components_.erase(components_.begin() + i + i,
        //                   components_.begin() + i + 2);
        for (size_t j = i + 1; j + 1 < components_.size(); j++)
          components_[j] = components_[j + 1];
        components_.pop_back();
        changed = true;
        num_collapsed++;
      }
    }
  }
  this->SetIndexes();
  this->Check();
  KALDI_LOG << "Collapsed " << num_collapsed << " components."
            << (num_collapsed == 0 && match_updatableness == true ?
                "  Try --match-updatableness=false." : "");
}

Nnet *GenRandomNnet(int32 input_dim,
                    int32 output_dim) {
  std::vector<Component*> components;
  int32 cur_dim = input_dim;
  // have up to 10 layers before the final one.
  for (size_t i = 0; i < 10; i++) {
    if (rand() % 2 == 0) {
      // add an affine component.
      int32 next_dim = 50 + rand() % 100;
      BaseFloat learning_rate = 0.0001, param_stddev = 0.001,
          bias_stddev = 0.1;
      AffineComponent *component = new AffineComponent();
      component->Init(learning_rate, cur_dim, next_dim,
                      param_stddev, bias_stddev);
      components.push_back(component);
      cur_dim = next_dim;
    } else if (rand() % 2 == 0) {
      components.push_back(new SigmoidComponent(cur_dim));
    } else if (rand() % 2 == 0 && cur_dim < 200) {
      SpliceComponent *component = new SpliceComponent();
      std::vector<int32> context;
      while (true) {
        context.clear();
        for (int32 i = -3; i <= 3; i++) {
          if (rand() % 3 == 0)
            context.push_back(i);
        }
        if (!context.empty() && context.front() <= 0 &&
            context.back() >= 0)
          break;
      }
      component->Init(cur_dim, context);
      components.push_back(component);
      cur_dim = cur_dim * context.size();
    } else {
      break;
    }
  }

  {
    AffineComponent *component = new AffineComponent();
    BaseFloat learning_rate = 0.0001, param_stddev = 0.001,
        bias_stddev = 0.1;
    component->Init(learning_rate, cur_dim, output_dim,
                    param_stddev, bias_stddev);
    components.push_back(component);
    cur_dim = output_dim;
  }

  components.push_back(new SoftmaxComponent(cur_dim));

  Nnet *ans = new Nnet();
  ans->Init(&components);
  return ans;
}

int32 Nnet::FirstUpdatableComponent() const {
  for (int32 i = 0; i < NumComponents(); i++) {
    if (dynamic_cast<UpdatableComponent*>(components_[i]) != NULL)
      return i;
  }
  return NumComponents();
}


int32 Nnet::LastUpdatableComponent() const {
  for (int32 i = NumComponents() - 1; i >= 0; i--)
    if (dynamic_cast<UpdatableComponent*>(components_[i]) != NULL)
      return i;
  return -1;
}

}  // namespace nnet2
}  // namespace kaldi

