ctc = [0.5]
pen = [1.5, 1.6, 1.7]

for i in ctc:
    for j in pen:
        fname = "conf/ja_grid/timesync_stctc" + str(i) + "_pen" + str(j) +"_beam50.yaml"
        with open(fname, "w") as f:
            conf = "batch_size: 1\nbeam_size: 50\npenalty: "+str(j)+"\nmaxlenratio: 0.0\nminlenratio: 0.0\nlm_weight: 0.0\nst_ctc_weight: "+str(i)+"\ntime_synchronous: True\nblank_penalty: 1.0"
            f.write(conf)

            cmd = "./run_ja.sh --st_config conf/tuning/encdec_hier_conformer_ja.yaml --stage 12 --stop_stage 13 --use_hier_ctc true --inference_config "+fname+" --inference_nj 200 --test_sets dev.en-ja"
            print(cmd)