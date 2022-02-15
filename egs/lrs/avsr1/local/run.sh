#!/bin/bash

chmod +x ./config.sh
. ./config.sh

MIN_SNR=-10
MAX_SNR=20
SNR_INTER=1

## See if BERT positional encoding improves performance
if [ "$VER" == 'mhanet-1.1c' ]
then
    python3 main.py --ver               $VER                        \
                    --network_type      'MHANetV3'                  \
                    --d_model           256                         \
                    --n_blocks          5                           \
                    --n_heads           8                           \
                    --warmup_steps      40000                       \
                    --causal            1                           \
                    --max_len           2048                       \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --outp_act          "Sigmoid"                   \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        200                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --map_type          'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           $MIN_SNR                    \
                    --max_snr           $MAX_SNR                    \
                    --snr_inter         $SNR_INTER                  \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi

if [ "$VER" == 'mhanet-1.0c' ]
then
    python3 main.py --ver               $VER                        \
                    --network_type      'MHANetV2'                  \
                    --d_model           256                         \
                    --n_blocks          5                           \
                    --n_heads           8                           \
                    --warmup_steps      40000                       \
                    --causal            1                           \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --outp_act          "Sigmoid"                   \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        200                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --map_type          'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           $MIN_SNR                    \
                    --max_snr           $MAX_SNR                    \
                    --snr_inter         $SNR_INTER                  \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi

# Updated ResNet with no scale and shift parameters for layer normalisation.
# This prevents overfitting to the training set.
if [ "$VER" == 'resnet-1.1c' ]
then
    python3 main.py --ver               $VER                        \
                    --network           'ResNetV2'                  \
                    --d_model           256                         \
                    --n_blocks          40                          \
                    --d_f               64                          \
                    --k                 3                           \
                    --max_d_rate        16                          \
                    --causal            1                           \
                    --unit_type         "ReLU->LN->W+b"             \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --outp_act          "Sigmoid"                   \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        200                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --map_type          'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           $MIN_SNR                    \
                    --max_snr           $MAX_SNR                    \
                    --snr_inter         $SNR_INTER                  \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi

# Updated ResNet with no scale and shift parameters for layer normalisation.
# This prevents overfitting to the training set.
if [ "$VER" == 'resnet-1.1n' ]
then
    python3 main.py --ver               $VER                        \
                    --network           'ResNetV2'                  \
                    --d_model           256                         \
                    --n_blocks          40                          \
                    --d_f               64                          \
                    --k                 3                           \
                    --max_d_rate        16                          \
                    --causal            0                           \
                    --unit_type         "ReLU->LN->W+b"             \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --outp_act          "Sigmoid"                   \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        180                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --map_type          'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           $MIN_SNR                    \
                    --max_snr           $MAX_SNR                    \
                    --snr_inter         $SNR_INTER                  \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi

if [ "$VER" == 'rdlnet-1.0n' ]
then
    python3 main.py --ver               $VER                        \
                    --network           'RDLNet'                    \
                    --n_blocks          18                          \
                    --length            7                           \
                    --m_1               64                          \
                    --causal            0                           \
                    --unit_type         "ReLU->scale*LN+center->W+b"\
                    --loss_fnc          "BinaryCrossentropy"        \
                    --outp_act          "Sigmoid"                   \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        180                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --map_type          'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           $MIN_SNR                    \
                    --max_snr           $MAX_SNR                    \
                    --snr_inter         $SNR_INTER                  \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi

if [ "$VER" == 'resnet-1.0c' ]
then
    python3 main.py --ver               'resnet-1.0c'   \
                    --network           'ResNet'        \
                    --d_model           256             \
                    --n_blocks          40              \
                    --d_f               64              \
                    --k                 3               \
                    --max_d_rate        16              \
                    --causal            1               \
                    --centre            1               \
                    --scale             1               \
                    --max_epochs        200             \
                    --resume_epoch      0               \
                    --test_epoch        100             \
                    --mbatch_size       8               \
                    --sample_size       1000            \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --outp_act          "Sigmoid"                   \
                    --inp_tgt_type      'MagXi'         \
                    --map_type          'DBNormalCDF'   \
                    --f_s               16000           \
                    --T_d               32              \
                    --T_s               16              \
                    --min_snr           $MIN_SNR        \
                    --max_snr           $MAX_SNR        \
                    --snr_inter         $SNR_INTER      \
                    --out_type          'y'             \
                    --save_model        1               \
                    --log_iter          0               \
                    --eval_example      0               \
                    --gain              $GAIN           \
                    --train             $TRAIN          \
                    --infer             $INFER          \
                    --test              $TEST           \
                    --gpu               $GPU            \
                    --set_path          $SET_PATH       \
                    --data_path         $DATA_PATH      \
                    --test_x_path       $TEST_X_PATH    \
                    --test_s_path       $TEST_S_PATH    \
                    --out_path          $OUT_PATH       \
                    --model_path        $MODEL_PATH
fi

if [ "$VER" == 'resnet-1.0n' ]
then
    python3 main.py --ver               'resnet-1.0n'   \
                    --network           'ResNet'        \
                    --d_model           256             \
                    --n_blocks          40              \
                    --d_f               64              \
                    --k                 3               \
                    --max_d_rate        16              \
                    --causal            0               \
                    --max_epochs        200             \
                    --resume_epoch      0               \
                    --test_epoch        180             \
                    --mbatch_size       8               \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --outp_act          "Sigmoid"                   \
                    --inp_tgt_type      'MagXi'         \
                    --map_type          'DBNormalCDF'   \
                    --sample_size       1000            \
                    --f_s               16000           \
                    --T_d               32              \
                    --T_s               16              \
                    --min_snr           $MIN_SNR        \
                    --max_snr           $MAX_SNR        \
                    --snr_inter         $SNR_INTER      \
                    --out_type          'y'             \
                    --save_model        1               \
                    --log_iter          0               \
                    --eval_example      0               \
                    --gain              $GAIN           \
                    --train             $TRAIN          \
                    --infer             $INFER          \
                    --test              $TEST           \
                    --gpu               $GPU            \
                    --set_path          $SET_PATH       \
                    --data_path         $DATA_PATH      \
                    --test_x_path       $TEST_X_PATH    \
                    --test_s_path       $TEST_S_PATH    \
                    --out_path          $OUT_PATH       \
                    --model_path        $MODEL_PATH
fi

if [ "$VER" == 'reslstm-1.0c' ]
then
    python3 main.py --ver               $VER                        \
                    --network           'ResLSTM'                   \
                    --d_model           512                         \
                    --n_blocks          5                           \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --outp_act          "Sigmoid"                   \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        200                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --map_type          'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           $MIN_SNR                    \
                    --max_snr           $MAX_SNR                    \
                    --snr_inter         $SNR_INTER                  \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi
