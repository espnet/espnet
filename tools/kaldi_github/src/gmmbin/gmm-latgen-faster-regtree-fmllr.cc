// gmmbin/gmm-latgen-faster-regtree-fmllr.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2013  Johns Hopkins University (author: Daniel Povey)
//                2014  Alpha Cephei Inc.

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "base/timer.h"
#include "transform/regression-tree.h"
#include "transform/regtree-fmllr-diag-gmm.h"
#include "transform/decodable-am-diag-gmm-regtree.h"
#include "feat/feature-functions.h"  // feature reversal

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using GMM-based model and RegTree-FMLLR adaptation.\n"
        "Usage: gmm-latgen-faster-regtree-fmllr [options] model-in regtree-in (fst-in|fsts-rspecifier) features-rspecifier transform-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;

    std::string word_syms_filename, utt2spk_rspecifier;
    config.Register(&po);
    po.Register("utt2spk", &utt2spk_rspecifier, "rspecifier for utterance to "
                "speaker map used to load the transform");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        regtree_in_str = po.GetArg(2),
        fst_in_str = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        xforms_rspecifier = po.GetArg(5),
        lattice_wspecifier = po.GetArg(6),
        words_wspecifier = po.GetOptArg(7),
        alignment_wspecifier = po.GetOptArg(8);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    RegressionTree regtree;
    {
      bool binary_read;
      Input in(regtree_in_str, &binary_read);
      regtree.Read(in.Stream(), binary_read, am_gmm);
    }

    RandomAccessRegtreeFmllrDiagGmmReaderMapped fmllr_reader(xforms_rspecifier,
                                                             utt2spk_rspecifier);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_done = 0, num_err = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      
      {
        LatticeFasterDecoder decoder(*decode_fst, config);
    
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          Matrix<BaseFloat> features (feature_reader.Value());
          feature_reader.FreeCurrent();
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_err++;
            continue;
          }
          if (!fmllr_reader.HasKey(utt)) {
            KALDI_WARN << "Not decoding utterance " << utt
                       << " because no transform available.";
            num_err++;
            continue;
          }

          RegtreeFmllrDiagGmm fmllr(fmllr_reader.Value(utt));

          kaldi::DecodableAmDiagGmmRegtreeFmllr gmm_decodable(am_gmm, trans_model,
                                                            features, fmllr,
                                                            regtree,
                                                            acoustic_scale);
          double like;
          if (DecodeUtteranceLatticeFaster(
                  decoder, gmm_decodable, trans_model, word_syms, utt, acoustic_scale,
                  determinize, allow_partial, &alignment_writer, &words_writer,
                  &compact_lattice_writer, &lattice_writer, &like)) {
            tot_like += like;
            frame_count += features.NumRows();
            num_done++;
          } else num_err++;
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);          
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }
        if (!fmllr_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no transform available.";
          num_err++;
          continue;
        }

        RegtreeFmllrDiagGmm fmllr(fmllr_reader.Value(utt));
        kaldi::DecodableAmDiagGmmRegtreeFmllr gmm_decodable(am_gmm, trans_model,
                                                            features, fmllr,
                                                            regtree,
                                                            acoustic_scale);

        LatticeFasterDecoder decoder(fst_reader.Value(), config);
        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, gmm_decodable, trans_model, word_syms, utt, acoustic_scale,
                determinize, allow_partial, &alignment_writer, &words_writer,
                &compact_lattice_writer, &lattice_writer, &like)) {
          tot_like += like;
          frame_count += features.NumRows();
          num_done++;
        } else num_err++;
      }
    }
      
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_done << " utterances, failed for "
              << num_err;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count << " frames.";

    delete word_syms;
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
