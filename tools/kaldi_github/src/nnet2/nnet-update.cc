// nnet2/nnet-update.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)
//           2014   Xiaohui Zhang

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

#include "nnet2/nnet-update.h"

namespace kaldi {
namespace nnet2 {



NnetUpdater::NnetUpdater(const Nnet &nnet,
                         Nnet *nnet_to_update):
    nnet_(nnet), nnet_to_update_(nnet_to_update) {
}
 


void NnetUpdater::FormatInput(const std::vector<NnetExample> &data) {

  forward_data_.resize(nnet_.NumComponents() + 1);
  Matrix<BaseFloat> input;
  FormatNnetInput(nnet_, data, &input);
  forward_data_[0].Resize(0, 0);  // avoids the next command ever copying GPU->CPU
  forward_data_[0].Swap(&input); // Copy to GPU, if being used.
  nnet_.ComputeChunkInfo(1 + nnet_.LeftContext() + nnet_.RightContext(),
                         data.size(), &chunk_info_out_);
}

double NnetUpdater::ComputeForMinibatch(
    const std::vector<NnetExample> &data,
    double *tot_accuracy) {

  FormatInput(data);
  Propagate();
  CuMatrix<BaseFloat> tmp_deriv;
  double ans = ComputeObjfAndDeriv(data, &tmp_deriv, tot_accuracy);
  if (nnet_to_update_ != NULL)
    Backprop(&tmp_deriv); // this is summed (after weighting), not
                          // averaged.
  return ans;
}


// form of ComputeForMinibatch for when the input data has
// already been formatted as a single matrix.
double NnetUpdater::ComputeForMinibatch(const std::vector<NnetExample> &data,
                                        Matrix<BaseFloat> *formatted_data,
                                        double *tot_accuracy) {
  { // accept the formatted input.  This replaces the call to FormatInput().
    int32 num_chunks = data.size();
    KALDI_ASSERT(formatted_data->NumRows() ==
                 num_chunks * (1 + nnet_.LeftContext() + nnet_.RightContext()) &&
                 formatted_data->NumCols() == nnet_.InputDim());

    forward_data_.resize(nnet_.NumComponents() + 1);
    // the next command avoids the Swap() command ever copying GPU->CPU in case
    // an instance of this class is used more than once (which it isn't in
    // practice).
    forward_data_[0].Resize(0, 0);  
    forward_data_[0].Swap(formatted_data); // Copy to GPU, if being used.
    nnet_.ComputeChunkInfo(1 + nnet_.LeftContext() + nnet_.RightContext(),
                           data.size(), &chunk_info_out_);
  }
  Propagate();
  CuMatrix<BaseFloat> tmp_deriv;
  double ans = ComputeObjfAndDeriv(data, &tmp_deriv, tot_accuracy);
  if (nnet_to_update_ != NULL)
    Backprop(&tmp_deriv); // this is summed (after weighting), not
                          // averaged.
  return ans;
}


void NnetUpdater::GetOutput(CuMatrix<BaseFloat> *output) {
  int32 num_components = nnet_.NumComponents(); 
  KALDI_ASSERT(forward_data_.size() == nnet_.NumComponents() + 1); 
  *output = forward_data_[num_components];
}

void NnetUpdater::Propagate() {
  static int32 num_times_printed = 0;
        
  int32 num_components = nnet_.NumComponents();
  for (int32 c = 0; c < num_components; c++) {
    const Component &component = nnet_.GetComponent(c);
    const CuMatrix<BaseFloat> &input = forward_data_[c];
    CuMatrix<BaseFloat> &output = forward_data_[c+1];
    // Note: the Propagate function will automatically resize the
    // output.
    component.Propagate(chunk_info_out_[c], chunk_info_out_[c+1], input, &output);
    // If we won't need the output of the previous layer for
    // backprop, delete it to save memory.
    bool need_last_output =
        (c>0 && nnet_.GetComponent(c-1).BackpropNeedsOutput()) ||
        component.BackpropNeedsInput();
    if (g_kaldi_verbose_level >= 3 && num_times_printed < 100) {
      KALDI_VLOG(3) << "Stddev of data for component " << c
                    << " for this minibatch is "
                    << (TraceMatMat(forward_data_[c], forward_data_[c], kTrans) /
                        (forward_data_[c].NumRows() * forward_data_[c].NumCols()));
      num_times_printed++;
    }
    if (!need_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data.
  }
}

double NnetUpdater::ComputeObjfAndDeriv(
    const std::vector<NnetExample> &data,
    CuMatrix<BaseFloat> *deriv,
    double *tot_accuracy) const {
  BaseFloat tot_objf = 0.0, tot_weight = 0.0;
  int32 num_components = nnet_.NumComponents();
  int32 num_chunks = data.size();
  deriv->Resize(num_chunks, nnet_.OutputDim()); // sets to zero.
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  KALDI_ASSERT(SameDim(output, *deriv));

  std::vector<MatrixElement<BaseFloat> > sv_labels;
  sv_labels.reserve(num_chunks); // We must have at least this many labels.
  for (int32 m = 0; m < num_chunks; m++) {
    KALDI_ASSERT(data[m].labels.size() == 1 &&
                 "Training code currently does not support multi-frame egs");
    const std::vector<std::pair<int32,BaseFloat> > &labels = data[m].labels[0];
    for (size_t i = 0; i < labels.size(); i++) {
      KALDI_ASSERT(labels[i].first < nnet_.OutputDim() &&
                        "Possibly egs come from alignments from mismatching model");
      MatrixElement<BaseFloat> elem = {m, labels[i].first, labels[i].second};
      sv_labels.push_back(elem);
    }
  }

  if (tot_accuracy != NULL)
    *tot_accuracy = ComputeTotAccuracy(data);
  
  deriv->CompObjfAndDeriv(sv_labels, output, &tot_objf, &tot_weight);
  
  KALDI_VLOG(4) << "Objective function is " << (tot_objf/tot_weight) << " over "
                << tot_weight << " samples (weighted).";
  return tot_objf;
}


double NnetUpdater::ComputeTotAccuracy(
    const std::vector<NnetExample> &data) const {
  BaseFloat tot_accuracy = 0.0;
  int32 num_components = nnet_.NumComponents();
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  KALDI_ASSERT(output.NumRows() == static_cast<int32>(data.size()));
  CuArray<int32> best_pdf(output.NumRows());
  std::vector<int32> best_pdf_cpu;
  
  output.FindRowMaxId(&best_pdf);
  best_pdf.CopyToVec(&best_pdf_cpu);

  for (int32 i = 0; i < output.NumRows(); i++) {
    KALDI_ASSERT(data[i].labels.size() == 1 &&
                 "Training code currently does not support multi-frame egs");
    const std::vector<std::pair<int32,BaseFloat> > &labels = data[i].labels[0];
    for (size_t j = 0; j < labels.size(); j++) {
      int32 ref_pdf_id = labels[j].first,
          hyp_pdf_id = best_pdf_cpu[i];
      BaseFloat weight = labels[j].second;
      tot_accuracy += weight * (hyp_pdf_id == ref_pdf_id ? 1.0 : 0.0);
    }
  }
  return tot_accuracy;
}


void NnetUpdater::Backprop(CuMatrix<BaseFloat> *deriv) const {
  // We assume ComputeObjfAndDeriv has already been called.
  for (int32 c = nnet_.NumComponents() - 1;
       c >= nnet_.FirstUpdatableComponent(); c--) {
    const Component &component = nnet_.GetComponent(c);
    Component *component_to_update = (nnet_to_update_ == NULL ? NULL :
                                      &(nnet_to_update_->GetComponent(c)));
    const CuMatrix<BaseFloat> &input = forward_data_[c],
        &output = forward_data_[c+1];
    CuMatrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());
    const CuMatrix<BaseFloat> &output_deriv(*deriv);
    component.Backprop(chunk_info_out_[c], chunk_info_out_[c+1], input, output,                       
                       output_deriv, component_to_update,
                       &input_deriv);
    input_deriv.Swap(deriv);
  }
}


void FormatNnetInput(const Nnet &nnet,
                     const std::vector<NnetExample> &data,
                     Matrix<BaseFloat> *input_mat) {
  KALDI_ASSERT(data.size() > 0);
  int32 num_splice = 1 + nnet.RightContext() + nnet.LeftContext();
  KALDI_ASSERT(data[0].input_frames.NumRows() >= num_splice);
  
  int32 feat_dim = data[0].input_frames.NumCols(),
         spk_dim = data[0].spk_info.Dim(),
         tot_dim = feat_dim + spk_dim; // we append these at the neural net
                                       // input... note, spk_dim might be 0.
  KALDI_ASSERT(tot_dim == nnet.InputDim());
  KALDI_ASSERT(data[0].left_context >= nnet.LeftContext());
  int32 ignore_frames = data[0].left_context - nnet.LeftContext(); // If
  // the NnetExample has more left-context than we need, ignore some.
  // this may happen in settings where we increase the amount of context during
  // training, e.g. by adding layers that require more context.  

  int32 num_chunks = data.size();
  
  input_mat->Resize(num_splice * num_chunks,
                    tot_dim, kUndefined);
  
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    SubMatrix<BaseFloat> dest(*input_mat,
                              chunk * num_splice, num_splice,
                              0, feat_dim);

    Matrix<BaseFloat> full_src(data[chunk].input_frames);
    SubMatrix<BaseFloat> src(full_src, ignore_frames, num_splice, 0, feat_dim);
                             
    dest.CopyFromMat(src);
    if (spk_dim != 0) {
      SubMatrix<BaseFloat> spk_dest(*input_mat,
                                    chunk * num_splice, num_splice,
                                    feat_dim, spk_dim);
      spk_dest.CopyRowsFromVec(data[chunk].spk_info);
    }
  }
}

BaseFloat TotalNnetTrainingWeight(const std::vector<NnetExample> &egs) {
  double ans = 0.0;
  for (size_t i = 0; i < egs.size(); i++)
    for (size_t j = 0; j < egs[i].labels.size(); j++) // for each labeled frame
      for (size_t k = 0; k < egs[i].labels[j].size(); k++)
        ans += egs[i].labels[j][k].second;
  return ans;
}


double ComputeNnetObjf(const Nnet &nnet,
                       const std::vector<NnetExample> &examples,
                       double *tot_accuracy) {
  NnetUpdater updater(nnet, NULL);
  return updater.ComputeForMinibatch(examples, tot_accuracy);
}

double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetExample> &examples,
                  Nnet *nnet_to_update,
                  double *tot_accuracy) {
  if (nnet_to_update == NULL)
    return ComputeNnetObjf(nnet, examples, tot_accuracy);
  try {
    NnetUpdater updater(nnet, nnet_to_update);
    return updater.ComputeForMinibatch(examples, tot_accuracy);
  } catch (...) {
    KALDI_LOG << "Error doing backprop, nnet info is: " << nnet.Info();
    throw;
  }
}

// version of DoBackprop that takes already-formatted examples.
double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetExample> &examples,
                  Matrix<BaseFloat> *examples_formatted,
                  Nnet *nnet_to_update,
                  double *tot_accuracy) {
  if (nnet_to_update == NULL) {
    KALDI_WARN << "Was not expecting to reach this code path "
               << "(wastefully formatting data twice)";
    return ComputeNnetObjf(nnet, examples, tot_accuracy);
 } try {
    NnetUpdater updater(nnet, nnet_to_update);
    return updater.ComputeForMinibatch(examples,
                                       examples_formatted,
                                       tot_accuracy);
  } catch (...) {
    KALDI_LOG << "Error doing backprop, nnet info is: " << nnet.Info();
    throw;
  }
}


double ComputeNnetGradient(
    const Nnet &nnet,
    const std::vector<NnetExample> &validation_set,
    int32 batch_size,
    Nnet *gradient) {
  bool treat_as_gradient = true;
  gradient->SetZero(treat_as_gradient);
  std::vector<NnetExample> batch;
  batch.reserve(batch_size);
  double tot_objf = 0.0;
  for (int32 start_pos = 0;
       start_pos < static_cast<int32>(validation_set.size());
       start_pos += batch_size) {
    batch.clear();
    for (int32 i = start_pos;
         i < std::min(start_pos + batch_size,
                      static_cast<int32>(validation_set.size()));
         i++) {
      batch.push_back(validation_set[i]);
    }
    tot_objf += DoBackprop(nnet,
                           batch,
                           gradient);
  }
  return tot_objf / validation_set.size();
}

double ComputeNnetObjf(
    const Nnet &nnet,
    const std::vector<NnetExample> &validation_set,
    int32 batch_size,
    double *tot_accuracy) {
  double tot_accuracy_tmp;
  if (tot_accuracy)
    *tot_accuracy = 0.0;
  std::vector<NnetExample> batch;
  batch.reserve(batch_size);
  double tot_objf = 0.0;
  for (int32 start_pos = 0;
       start_pos < static_cast<int32>(validation_set.size());
       start_pos += batch_size) {
    batch.clear();
    for (int32 i = start_pos;
         i < std::min(start_pos + batch_size,
                      static_cast<int32>(validation_set.size()));
         i++) {
      batch.push_back(validation_set[i]);
    }
    tot_objf += ComputeNnetObjf(nnet, batch,
                                tot_accuracy != NULL ? &tot_accuracy_tmp : NULL);
    if (tot_accuracy)
      *tot_accuracy += tot_accuracy_tmp;
  }
  return tot_objf;
}

  
  
} // namespace nnet2
} // namespace kaldi
