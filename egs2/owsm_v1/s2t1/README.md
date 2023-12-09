### Guidance for data preparation
(1) Please work progressively from v1 to v3: this means you need to prepare data for v1, v2 and v3 in order to obtain the full v3 data. To start the data preparation, run `bash local/data.sh --VERSION v1 # or v2, v3`
(2) Please revise `db.sh` for all datasets before running `local/data.sh`. Some datasets cannot be downloaded and untared automatically due to license issues. Users should take care of it by themselves.
(3) Due to the large volume of data, we are not confident the scripts will run smoothly for each dataset. Please raise an issue if you believe there is a bug.
(4) This script only prepares data for train and valid subsets. Test data should be prepared separately following the conventional Espnet2 format.
(5) Even though we provide this centralized data preparation script and combine all datasets in it, we strongly recommend users to NOT use the merged train_v* and valid_v* for feature extractions. Instead, users may run stage 2-4 for each dataset separately and combine all datasets together under `dump/raw` directory. This will allow you to handle all datasets simultaneously; inspection and debugging will also be easier. This is exactly what we did in our experiments.
(6) Users can also refer to this PR to check more details: https://github.com/espnet/espnet/pull/5478
(7) The detailed data list is in `local/data.sh`. Also see: https://arxiv.org/pdf/2309.13876.pdf
