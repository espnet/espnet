// nnet2bin/nnet-normalize-stddev.cc

// Copyright 2013  Guoguo Chen
//           2014  Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet2/train-nnet.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program first identifies any affine or block affine layers that\n"
        "are followed by pnorm and then renormalize layers. Then it rescales\n"
        "those layers such that the parameter stddev is 1.0 after scaling\n"
        "(the target stddev is configurable by the --stddev option).\n"
        "If you supply the option --stddev-from=<model-filename>, it rescales\n"
        "those layers to match the standard deviations of corresponding layers\n"
        "in the specified model.\n"
        "\n"
        "Usage: nnet-normalize-stddev [options] <model-in> <model-out>\n"
        " e.g.: nnet-normalize-stddev final.mdl final.mdl\n";

    bool binary_write = true;
    BaseFloat stddev = 1.0;
    std::string reference_model_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("stddev-from", &reference_model_filename, "Reference model");
    po.Register("stddev", &stddev, "Target standard deviation that we normalize "
                "to (note: is overridden by --stddev-from option, if supplied)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        normalized_nnet_rxfilename = po.GetArg(2);

    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    int32 ret = 0;

    // Works out the layers that we would like to normalize: any affine or block
    // affine layers that are followed by pnorm and then renormalize layers.
    std::vector<int32> identified_components;
    for (int32 c = 0; c < am_nnet.GetNnet().NumComponents() - 2; c++) {
      // Checks if the current layer is an affine layer or block affine layer.
      // Also includes PreconditionedAffineComponent and
      // PreconditionedAffineComponentOnline, since they are child classes of
      // AffineComponent.
      kaldi::nnet2::Component *component = &(am_nnet.GetNnet().GetComponent(c));
      AffineComponent *ac = dynamic_cast<AffineComponent*>(component);
      BlockAffineComponent *bac =
        dynamic_cast<BlockAffineComponent*>(component);
      if (ac == NULL && bac == NULL)
        continue;

      // Checks if the next layer is a pnorm layer.
      component = &(am_nnet.GetNnet().GetComponent(c + 1));
      PnormComponent *pc = dynamic_cast<PnormComponent*>(component);
      if (pc == NULL)
        continue;

      // Checks if the layer after the pnorm layer is a NormalizeComponent
      // or a PowerComponent followed by a NormalizeComponent
      component = &(am_nnet.GetNnet().GetComponent(c + 2));
      NormalizeComponent *nc = dynamic_cast<NormalizeComponent*>(component);
      PowerComponent *pwc = dynamic_cast<PowerComponent*>(component);
      if (nc == NULL && pwc == NULL)
        continue;
      if (pwc != NULL) {  // verify it's PowerComponent followed by
                         // NormalizeComponent.
        if (c + 3 >= am_nnet.GetNnet().NumComponents())
          continue;
        component = &(am_nnet.GetNnet().GetComponent(c + 3));
        nc = dynamic_cast<NormalizeComponent*>(component);
        if (nc == NULL)
          continue;
      }
      // This is the layer that we would like to normalize.
      identified_components.push_back(c);
    }

    AmNnet am_nnet_ref;
    if (!reference_model_filename.empty()) {
      bool binary_read;
      Input ki(reference_model_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet_ref.Read(ki.Stream(), binary_read);
      KALDI_ASSERT(am_nnet_ref.GetNnet().NumComponents() == am_nnet.GetNnet().NumComponents());
    }

    BaseFloat ref_stddev = 0.0;

    // Normalizes the identified layers.
    for (int32 c = 0; c < identified_components.size(); c++) {
      ref_stddev = stddev;
      if (!reference_model_filename.empty()) {
        kaldi::nnet2::Component *component =
            &(am_nnet_ref.GetNnet().GetComponent(identified_components[c]));
        UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(component);
        KALDI_ASSERT(uc != NULL);
        Vector<BaseFloat> params(uc->GetParameterDim());
        uc->Vectorize(&params);
        BaseFloat params_average = params.Sum()
            / static_cast<BaseFloat>(params.Dim());
        params.Add(-1.0 * params_average);
        ref_stddev = sqrt(VecVec(params, params)
            / static_cast<BaseFloat>(params.Dim()));
      }

      kaldi::nnet2::Component *component =
          &(am_nnet.GetNnet().GetComponent(identified_components[c]));
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(component);
      KALDI_ASSERT(uc != NULL);
      Vector<BaseFloat> params(uc->GetParameterDim());
      uc->Vectorize(&params);
      BaseFloat params_average = params.Sum()
          / static_cast<BaseFloat>(params.Dim());
      params.Add(-1.0 * params_average);
      BaseFloat params_stddev = sqrt(VecVec(params, params)
          / static_cast<BaseFloat>(params.Dim()));
      if (params_stddev > 0.0) {
        uc->Scale(ref_stddev / params_stddev);
        KALDI_LOG << "Normalized component " << identified_components[c];
      }
    }

    // Writes the normalized model.
    Output ko(normalized_nnet_rxfilename, binary_write);
    trans_model.Write(ko.Stream(), binary_write);
    am_nnet.Write(ko.Stream(), binary_write);

    return ret;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
