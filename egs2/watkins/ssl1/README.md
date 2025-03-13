# Marine Sound Classification Performance Report

These experiments analyze the impact of temporal ratio of "interesting events" to noise in the pretraining corpus for BEATs pre-training.

We compare two pre-trained model variants - all_cuts (base dataset) and all_cuts_noise (with augmented noise) - each on two variants differentiated by tokenizer initialization. 

Results indicate that the choice of tokenizer significantly affects model performance and a bad temporal ratio leads to an underperforming model (especially if the tokenizer is not well trained).

## Dataset details

- **Base Dataset (all_cuts)**: Original marine sounds collection (whales, etc.) scraped from "all cut" section of Watkins website (https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm). This is ~30 hours of data

- **Augmented Dataset (all_cuts_noise)**: Constructed by clipping portions from the base dataset and appending random noise [N(0,0.01)] on both ends (ensuring none of the original audio is left out, for fairness.) This is 123 hours total (where 30 hours is original data and 93 hours is noise).

- **Sample Duration**: All examples in the augmented dataset were standardized to 10 seconds, while the original dataset contained variable-length clips

* Note that we do not use master tapes (as we discussed in last meeting) instead of augmented dataset, because I listened to them and they contain speech along with the marine sounds (and it is hard to separate speech out due to lack of relevant metadata). Moreover, this was a cleaner setup (equal data, just changing temporal ratio for testing our hypothesis).

- **Evaluation Dataset**: Watkins (standard BEANS benchmark classification task), containing 31 different aquatic mammals. All models were then fully fine-tuned on the downstream task. The test set has 331 samples.

## Results

### Full fine-tuning

| Model Name        | Tokenizer Type       | Test Mean Acc (%) | Pre-training corpus | Pre-training steps |
|-------------------|----------------------|-------------------| --------------------| -- |
| all_cuts         | Random               | 85.55             | Watkins cut tapes   | 100k |
| all_cuts         | BEATs (iter 3)       | 87.91             | Watkins cut tapes   | 100k |
| all_cuts_noise   | Random               | 78.76             | Watkins cut tapes with noise   | 100k |
| all_cuts_noise   | BEATs (iter 3)       | 89.09             | Watkins cut tapes with noise   | 100k |
| BEATs iter 3      | BEATs (iter 2)       | 89.40              | Audioset   | 400k |

### Linear probing

| Model Name        | Tokenizer Type       | Test Mean Acc (%) | Pre-training corpus | Pre-training steps |
|-------------------|----------------------|-------------------| --------------------| --------------------|
| all_cuts         | Random               | 51.62            | Watkins cut tapes   | 100k |
| all_cuts         | BEATs (iter 3)       | 73.16             | Watkins cut tapes   | 100k |
| all_cuts_noise   | Random               | 33.04             | Watkins cut tapes with noise   | 100k |
| all_cuts_noise   | BEATs (iter 3)       | 64.31             | Watkins cut tapes with noise   | 100k |



### Model Variations

- **Tokenizer: Random**: Training targets are generated via a random tokenizer (BEST-RQ style)
- **Tokenizer: BEATs iteration 3**: This is a self-distilled tokenizer created from BEATs iteration 3 encoder.
- **All encoders** were trained from scratch

## Training Parameters

### Configuration

- **Training Steps**: 100,000 steps for all models
- **Optimizer**: AdamW
- **Learning Rate**: 5.0e-4
- **Batch Size**: Average of 26 examples
- **Model Architecture**: BEATs (Vision Transformer-like)
- **Warmup Steps**: 
  - 10,000 steps for all models except,
  - 40,000 steps for all_cuts with random tokenizer (10,000 leads to collapse of pre-training)

### Evaluation Protocol

- **Evaluation Point**: At 100,000 steps (selected based on convergence of pre-trained model)
- **Metric**: Classification accuracy

## Key Findings

1. **Tokenizer Impact**:
   - Models using the BEATs iteration 3 tokenizer consistently outperformed those with random tokenizers
   - The performance gap between tokenizers was substantially larger for the noisy dataset

2. **Noise Effect**:
   - With a random tokenizer, adding noise degraded performance (85.6 to 78.8)
   - With the BEATs tokenizer, the impact of adding noise is almost insignificant (slightly positive even but the test set size is 331)

3. **Best Configuration**:
   - The all_cuts_noise model with the BEATs tokenizer (iteration 1) achieved the highest overall accuracy 89.09% very close to the BEATs iter 3 model pre-trained on just Audioset (89.4).

### Other details

- The all_cuts model with random tokenizer (iteration 0) collapsed to a single token with the standard 10,000-step warmup
- Increasing warmup to 40,000 steps resolved this issue
- No similar issues were encountered with any other models we pre-trained.
