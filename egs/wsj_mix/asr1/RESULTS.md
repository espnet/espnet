# WSJ-2mix Result
## word level rnnlm with / without speaker parallel attention
## CER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_vggblstmp_sde1_rece2_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_spafalse_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs10_mli1000_mlo150_lsmunigram0.05_delta/decode_cv_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000 | 503 | 85377 | 6581 | 4890 | 3982 | 15.96 |
| exp/tr_pytorch_vggblstmp_sde1_rece2_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_spafalse_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs10_mli1000_mlo150_lsmunigram0.05_delta/decode_tt_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000 | 333 | 60849 | 3537 | 2695 | 1920 | 12.15 |
| exp/tr_pytorch_vggblstmp_sde1_rece2_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_spatrue_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs10_mli800_mlo150_lsmunigram0.05_delta/decode_cv_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000 | 503 | 85388 | 5927 | 5533 | 2875 | 14.80 |
| exp/tr_pytorch_vggblstmp_sde1_rece2_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_spatrue_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs10_mli800_mlo150_lsmunigram0.05_delta/decode_tt_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000 | 333 | 61630 | 3176 | 2275 | 1842 | 10.87 |
## WER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_vggblstmp_sde1_rece2_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_spafalse_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs10_mli1000_mlo150_lsmunigram0.05_delta/decode_cv_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000 | 503 | 12691 | 3169 | 566 | 651 | 26.70 |
| exp/tr_pytorch_vggblstmp_sde1_rece2_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_spafalse_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs10_mli1000_mlo150_lsmunigram0.05_delta/decode_tt_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000 | 333 | 9350 | 1677 | 291 | 308 | 20.11 |
| exp/tr_pytorch_vggblstmp_sde1_rece2_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_spatrue_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs10_mli800_mlo150_lsmunigram0.05_delta/decode_cv_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000 | 503 | 12817 | 2921 | 688 | 475 | 24.86 |
| exp/tr_pytorch_vggblstmp_sde1_rece2_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_spatrue_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs10_mli800_mlo150_lsmunigram0.05_delta/decode_tt_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000 | 333 | 9557 | 1501 | 260 | 308 | 18.28 |

The mixture scheme is in the local/mix_2_spk_max_{tr,cv,tt}_mix
Click here to get the [pretrained model without speaker parallel attention](https://drive.google.com/open?id=11SWTPG5ggMHtqucHDTeWpNCRXrYMw4SZ).
Click here to get the [pretrained model with speaker parallel attention]().

# WSJ0-2mix
# future work
