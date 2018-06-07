// tree/context-dep.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "tree/context-dep.h"
#include "base/kaldi-math.h"
#include "tree/build-tree.h"

namespace kaldi {

bool ContextDependency::Compute(const std::vector<int32> &phoneseq,
                                 int32 pdf_class,
                                 int32 *pdf_id) const {
  KALDI_ASSERT(static_cast<int32>(phoneseq.size()) == N_);
  EventType  event_vec;
  event_vec.reserve(N_+1);
  event_vec.push_back(std::make_pair
                      (static_cast<EventKeyType>(kPdfClass),  // -1
                       static_cast<EventValueType>(pdf_class)));
  KALDI_COMPILE_TIME_ASSERT(kPdfClass < 0);  // or it would not be sorted.
  for (int32 i = 0;i < N_;i++) {
    event_vec.push_back(std::make_pair
                        (static_cast<EventKeyType>(i),
                         static_cast<EventValueType>(phoneseq[i])));
    KALDI_ASSERT(static_cast<EventAnswerType>(phoneseq[i]) >= 0);
  }
  KALDI_ASSERT(pdf_id != NULL);
  return to_pdf_->Map(event_vec, pdf_id);
}

ContextDependency *GenRandContextDependency(const std::vector<int32> &phone_ids,
                                            bool ensure_all_covered,
                                            std::vector<int32> *hmm_lengths) {
  KALDI_ASSERT(IsSortedAndUniq(phone_ids));
  int32 num_phones = phone_ids.size();
  int32 num_stats = 1 + (Rand() % 15) * (Rand() % 15);  // up to 14^2 + 1 separate stats.
  int32 N = 2 + Rand() % 3;  // 2, 3 or 4.
  int32 P = Rand() % N;
  float ctx_dep_prob = 0.7 + 0.3*RandUniform();
  int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
  hmm_lengths->clear();
  hmm_lengths->resize(max_phone + 1, -1);
  std::vector<bool> is_ctx_dep(max_phone + 1);

  for (int32 i = 0; i <= max_phone; i++) {
    (*hmm_lengths)[i] = 1 + Rand() % 3;
    is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
  }
  for (size_t i = 0; i < (size_t) num_phones; i++)
    KALDI_VLOG(2) <<  "For idx = " << i
                  << ", (phone_id, hmm_length, is_ctx_dep) == "
                  << (phone_ids[i]) << " " << ((*hmm_lengths)[phone_ids[i]])
                  << " " << (is_ctx_dep[phone_ids[i]]);
  // Generate rand stats.
  BuildTreeStatsType stats;
  size_t dim = 3 + Rand() % 20;
  GenRandStats(dim, num_stats, N, P, phone_ids, *hmm_lengths,
               is_ctx_dep, ensure_all_covered, &stats);

  // Now build the tree.

  Questions qopts;
  int32 num_quest = Rand() % 10, num_iters = rand () % 5;
  qopts.InitRand(stats, num_quest, num_iters, kAllKeysUnion);  // This was tested in build-tree-utils-test.cc

  float thresh = 100.0 * RandUniform();

  EventMap *tree = NULL;
  std::vector<std::vector<int32> > phone_sets(phone_ids.size());
  for (size_t i = 0; i < phone_ids.size(); i++)
    phone_sets[i].push_back(phone_ids[i]);
  std::vector<bool> share_roots(phone_sets.size(), true),
      do_split(phone_sets.size(), true);

  tree = BuildTree(qopts, phone_sets, *hmm_lengths, share_roots,
                   do_split, stats, thresh, 1000, 0.0, P);
  DeleteBuildTreeStats(&stats);
  return new ContextDependency(N, P, tree);
}


ContextDependency *GenRandContextDependencyLarge(const std::vector<int32> &phone_ids,
                                                 int N, int P,
                                                 bool ensure_all_covered,
                                                 std::vector<int32> *hmm_lengths) {
  KALDI_ASSERT(IsSortedAndUniq(phone_ids));
  int32 num_phones = phone_ids.size();
  int32 num_stats = 3000;  // each is a separate context.
  float ctx_dep_prob = 0.9;
  KALDI_ASSERT(num_phones > 0);
  hmm_lengths->clear();
  int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
  hmm_lengths->resize(max_phone + 1, -1);
  std::vector<bool> is_ctx_dep(max_phone + 1);

  for (int32 i = 0; i <= max_phone; i++) {
    (*hmm_lengths)[i] = 1 + Rand() % 3;
    is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
  }
  for (size_t i = 0; i < (size_t) num_phones; i++) {
    KALDI_VLOG(2) <<  "For idx = "<< i << ", (phone_id, hmm_length, is_ctx_dep) == " << (phone_ids[i]) << " " << ((*hmm_lengths)[phone_ids[i]]) << " " << (is_ctx_dep[phone_ids[i]]);
  }
  // Generate rand stats.
  BuildTreeStatsType stats;
  size_t dim = 3 + Rand() % 20;
  GenRandStats(dim, num_stats, N, P, phone_ids, *hmm_lengths, is_ctx_dep, ensure_all_covered, &stats);

  // Now build the tree.

  Questions qopts;
  int32 num_quest = 40, num_iters = 0;
  qopts.InitRand(stats, num_quest, num_iters, kAllKeysUnion);  // This was tested in build-tree-utils-test.cc

  float thresh = 100.0 * RandUniform();

  EventMap *tree = NULL;
  std::vector<std::vector<int32> > phone_sets(phone_ids.size());
  for (size_t i = 0; i < phone_ids.size(); i++)
    phone_sets[i].push_back(phone_ids[i]);
  std::vector<bool> share_roots(phone_sets.size(), true),
      do_split(phone_sets.size(), true);

  tree = BuildTree(qopts, phone_sets, *hmm_lengths, share_roots,
                   do_split, stats, thresh, 1000, 0.0, P);
  DeleteBuildTreeStats(&stats);
  return new ContextDependency(N, P, tree);
}


void ContextDependency::Write (std::ostream &os, bool binary) const {
  WriteToken(os, binary, "ContextDependency");
  WriteBasicType(os, binary, N_);
  WriteBasicType(os, binary, P_);
  WriteToken(os, binary, "ToPdf");
  to_pdf_->Write(os, binary);
  WriteToken(os, binary, "EndContextDependency");
}


void ContextDependency::Read (std::istream &is, bool binary) {
  if (to_pdf_) {
    delete to_pdf_;
    to_pdf_ = NULL;
  }
  ExpectToken(is, binary, "ContextDependency");
  ReadBasicType(is, binary, &N_);
  ReadBasicType(is, binary, &P_);
  EventMap *to_pdf = NULL;
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "ToLength") {  // back-compat.
    EventMap *to_num_pdf_classes = EventMap::Read(is, binary);
    delete to_num_pdf_classes;
    ReadToken(is, binary, &token);
  }
  if (token == "ToPdf") {
    to_pdf = EventMap::Read(is , binary);
  } else {
    KALDI_ERR << "Got unexpected token " << token
              << " reading context-dependency object.";
  }
  ExpectToken(is, binary, "EndContextDependency");
  to_pdf_ = to_pdf;
}

void ContextDependency::EnumeratePairs(
    const std::vector<int32> &phones,
    int32 self_loop_pdf_class, int32 forward_pdf_class,
    const std::vector<int32> &phone_window,
    unordered_set<std::pair<int32, int32>, PairHasher<int32> > *pairs) const {
  std::vector<int32> new_phone_window(phone_window);
  EventType vec;

  std::vector<EventAnswerType> forward_pdfs, self_loop_pdfs;

  // get list of possible forward pdfs
  vec.clear();
  for (size_t i = 0; i < N_; i++)
    if (phone_window[i] >= 0)
      vec.push_back(std::make_pair(static_cast<EventKeyType>(i),
                                   static_cast<EventValueType>(phone_window[i])));
  vec.push_back(std::make_pair(kPdfClass, static_cast<EventValueType>(forward_pdf_class)));
  std::sort(vec.begin(), vec.end());
  to_pdf_->MultiMap(vec, &forward_pdfs);
  SortAndUniq(&forward_pdfs);

  // get list of possible self-loop pdfs
  vec.clear();
  for (size_t i = 0; i < N_; i++)
    if (phone_window[i] >= 0)
      vec.push_back(std::make_pair(static_cast<EventKeyType>(i),
                                   static_cast<EventValueType>(phone_window[i])));
  vec.push_back(std::make_pair(kPdfClass, static_cast<EventValueType>(self_loop_pdf_class)));
  std::sort(vec.begin(), vec.end());
  to_pdf_->MultiMap(vec, &self_loop_pdfs);
  SortAndUniq(&self_loop_pdfs);

  if (forward_pdfs.size() == 1 || self_loop_pdfs.size() == 1) {
    for (size_t m = 0; m < forward_pdfs.size(); m++)
      for (size_t n = 0; n < self_loop_pdfs.size(); n++)
        pairs->insert(std::make_pair(forward_pdfs[m], self_loop_pdfs[n]));
  } else {
    // Choose 'position' as a phone position in 'context' that's currently
    // -1, and that is as close as possible to the central position P.
    int32 position = 0;
    int32 min_dist = N_ - 1;
    for (int32 i = 0; i < N_; i++) {
      int32 dist = (P_ - i > 0) ? (P_ - i) : (i - P_);
      if (phone_window[i] == -1 && dist < min_dist) {
        position = i;
        min_dist = dist;
      }
    }
    KALDI_ASSERT(min_dist < N_);
    KALDI_ASSERT(position != P_);

    // The next two lines have to do with how BOS/EOS effects are handled in
    // phone context.  Zero phone value in a non-central position (i.e. not
    // position P_...  and 'position' will never equal P_) means 'there is no
    // phone here because we're at BOS or EOS'.
    new_phone_window[position] = 0;
    EnumeratePairs(phones, self_loop_pdf_class, forward_pdf_class,
                   new_phone_window, pairs);

    for (size_t i = 0 ; i < phones.size(); i++) {
      new_phone_window[position] = phones[i];
      EnumeratePairs(phones, self_loop_pdf_class, forward_pdf_class,
                     new_phone_window, pairs);
    }
  }
}

void ContextDependency::GetPdfInfo(
    const std::vector<int32> &phones,
    const std::vector<std::vector<std::pair<int32, int32> > > &pdf_class_pairs,
    std::vector<std::vector<std::vector<std::pair<int32, int32> > > > *pdf_info) const {

  KALDI_ASSERT(pdf_info != NULL);
  pdf_info->resize(1 + *std::max_element(phones.begin(), phones.end()));
  std::vector<int32> phone_window(N_, -1);
  EventType vec;
  for (size_t i = 0 ; i < phones.size(); i++) {
    // loop over phones
    int32 phone = phones[i];
    (*pdf_info)[phone].resize(pdf_class_pairs[phone].size());
    for (size_t j = 0; j < pdf_class_pairs[phone].size(); j++) {
      // loop over pdf_class pairs
      int32 pdf_class = pdf_class_pairs[phone][j].first,
            self_loop_pdf_class = pdf_class_pairs[phone][j].second;
      phone_window[P_] = phone;

      unordered_set<std::pair<int32, int32>, PairHasher<int32> > pairs;
      EnumeratePairs(phones, self_loop_pdf_class, pdf_class, phone_window, &pairs);
      unordered_set<std::pair<int32, int32>, PairHasher<int32> >::iterator iter = pairs.begin(),
                           end = pairs.end();
      for (; iter != end; ++iter)
        (*pdf_info)[phone][j].push_back(*iter);
      std::sort( ((*pdf_info)[phone][j]).begin(),  ((*pdf_info)[phone][j]).end());
    }
  }
}

void ContextDependency::GetPdfInfo(
    const std::vector<int32> &phones,
    const std::vector<int32> &num_pdf_classes,  // indexed by phone,
    std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const {

  EventType vec;
  KALDI_ASSERT(pdf_info != NULL);
  pdf_info->resize(NumPdfs());
  for (size_t i = 0 ; i < phones.size(); i++) {
    int32 phone = phones[i];
    vec.clear();
    vec.push_back(std::make_pair(static_cast<EventKeyType>(P_),
                                 static_cast<EventValueType>(phone)));
    // Now get length.
    KALDI_ASSERT(static_cast<size_t>(phone) < num_pdf_classes.size());
    EventAnswerType len = num_pdf_classes[phone];

    for (int32 pos = 0; pos < len; pos++) {
      vec.resize(2);
      vec[0] = std::make_pair(static_cast<EventKeyType>(P_),
                              static_cast<EventValueType>(phone));
      vec[1] = std::make_pair(kPdfClass, static_cast<EventValueType>(pos));
      std::sort(vec.begin(), vec.end());
      std::vector<EventAnswerType> pdfs;  // pdfs that can be at this pos as this phone.
      to_pdf_->MultiMap(vec, &pdfs);
      SortAndUniq(&pdfs);
      if (pdfs.empty()) {
        KALDI_WARN << "ContextDependency::GetPdfInfo, no pdfs returned for position "<< pos << " of phone " << phone << ".   Continuing but this is a serious error.";
      }
      for (size_t j = 0; j < pdfs.size(); j++) {
        KALDI_ASSERT(static_cast<size_t>(pdfs[j]) < pdf_info->size());
        (*pdf_info)[pdfs[j]].push_back(std::make_pair(phone, pos));
      }
    }
  }
  for (size_t i = 0; i < pdf_info->size(); i++) {
    std::sort( ((*pdf_info)[i]).begin(),  ((*pdf_info)[i]).end());
    KALDI_ASSERT(IsSortedAndUniq( ((*pdf_info)[i])));  // should have no dups.
  }
}



ContextDependency*
MonophoneContextDependency(const std::vector<int32> phones,
                           const std::vector<int32> phone2num_pdf_classes) {
  std::vector<std::vector<int32> > phone_sets(phones.size());
  for (size_t i = 0; i < phones.size(); i++) phone_sets[i].push_back(phones[i]);
  std::vector<bool> share_roots(phones.size(), false);  // don't share roots.
  // N is context size, P = position of central phone (must be 0).
  int32 num_leaves = 0, P = 0, N = 1;
  EventMap *pdf_map = GetStubMap(P, phone_sets, phone2num_pdf_classes, share_roots, &num_leaves);
  return new ContextDependency(N, P, pdf_map);
}

ContextDependency*
MonophoneContextDependencyShared(const std::vector<std::vector<int32> > phone_sets,
                                 const std::vector<int32> phone2num_pdf_classes) {
  std::vector<bool> share_roots(phone_sets.size(), false);  // don't share roots.
  // N is context size, P = position of central phone (must be 0).
  int32 num_leaves = 0, P = 0, N = 1;
  EventMap *pdf_map = GetStubMap(P, phone_sets, phone2num_pdf_classes, share_roots, &num_leaves);
  return new ContextDependency(N, P, pdf_map);
}





} // end namespace kaldi.
