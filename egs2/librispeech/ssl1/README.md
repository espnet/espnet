## INTRODUCTION

This recipe trains a [Hubert](https://arxiv.org/pdf/2106.07447.pdf)pretrain model, using data Librispeech 960hr data, including the k-means-based pseudo label generation and mask-prediction training.

This recipe requires fairseq installed, please run:

    cd ${MAIN_ROOT}/tools && make fairseq.done

Run.sh calls hubert.sh, and there are 7 stages in total. First, we need to do some data preparation to build espnet-style data/dump folders(stage 0-4). 

    ./run.sh --stage 1 --stop-stage 4

Then stage 5 calls script/km.sh to generate pseudo labels used in pretraining. To run this stage, please specify:
    train/valid set, 
    number of k-means clusters, 
    feature type(could be mfcc or hubert extracted feature, will explain later), 
    the percentage you want to used to train the k-means model.
These parameters can be settled in run.sh and pass to hubert.sh

    ./run.sh --stage 5 --stop-stage 5
	
Stage 6 and 7 collect stats of train/valid set and train the hubert model respectively. These two stages have the same functionality as stage 10 and 11 of asr recipes. The only difference is we call `espnet2.bin.hubert_train` instead of `espnet2.bin.train` for training.

    ./run.sh --stage 6 --stop-stage 7
	
Note that stage 5-7 could run multiple times, which is specified by `pretrain_start_iter` and `pretrain_start_iter`. You may find the default value in run.sh:

    pretrain_start_iter=0
    pretrain_stop_iter=2
	
That refers to 3 iterations, n_clusters_iter[0-2] and feature_iter[0-2] specify the number of clusters and the feature type used for k-means clustering of different iterations. Follow the [Hubert](https://arxiv.org/pdf/2106.07447.pdf) settings of base model, we use mfcc and 100 clusters for the iteration 0, and extract the latent features from transformer layer 6 of HuBERT model(HuBERT6) pre-trained in previous iteration, 500 clusters for iteration 1, and (HuBERT9) 500 clusters for iteration 2. Each iteration has a different config file. Please refer to conf/tuning/train_asr_hubert_base_960h_pretrain_it*.yaml

This is the end of Hubert pretraining. After the pretraining finish, you can run the finetuning stage with run.sh under any asr recipes. An example finetuning config file is egs2/librilight_limited/asr1/conf/tuning/train_asr_hubert_base_10h_finetuning.yaml:

    cd ../../librilight_limited/asr1/
	./run.sh

================================================

## RESULTS

The `CER` and `WER` in following result is got after librilight_limited-10hr finetuning:

### iteration 0 without language model:
- Model files
    - model link: (TO BE ADDED)
    - training config file: `conf/tuning/train_asr_hubert_base_960h_pretrain_it0.yaml` 
    - e2e file: `exp/pretrain_train_asr_hubert_base_960h_pretrain_it0_raw_iter0/valid.acc.best.pth`    
    - e2e JSON file: `exp/pretrain_train_asr_hubert_base_960h_pretrain_it0_raw_iter0/config.yaml`    
  - Results
  (TO BE ADDED)

  (MORE RESULT PENDING TO BE ADDED)
  
### iteration 0 with 4-gram language model:
### iteration 0 without language model:
### iteration 1 with 4-gram language model:

================================================

## HUBERT IN FAIRSEQ

The original Hubert paper, code and model can be found in:
paper: https://arxiv.org/pdf/2106.07447.pdf
code and model: https://github.com/pytorch/fairseq/tree/master/examples/hubert

================================================

## ACKNOWLEDGEMENT

We would like to thank Wei-Ning Hsu(Facebook) and Abdelrahman Mohamed(Facebook) for their work on Hubert and valuable
information/kind help of this implementation.