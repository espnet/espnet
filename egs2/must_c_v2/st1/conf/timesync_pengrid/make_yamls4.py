ctc = [0.5]
beam = [30]
bpen=[0.8,0.6]
pen = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

for i in ctc:
    for b in beam:
        for bp in bpen:
            for j in pen:
            
                fname = "conf/timesync_pengrid/timesync" + str(i) + "_pen" + str(j) +"_beam"+str(b)+"_blankpen"+str(bp)+".yaml"
                with open(fname, "w") as f:
                    conf = "batch_size: 1\nbeam_size: "+str(b)+"\npenalty: "+str(j)+"\nmaxlenratio: 0.0\nminlenratio: 0.0\nlm_weight: 0.0\nst_ctc_weight: "+str(i)+"\ntime_synchronous: True\nblank_penalty: "+str(bp)+"\n"
                    f.write(conf)

                    cmd = "./run.sh --st_config conf/tuning/encdec_hier_conformer.yaml --stage 12 --stop_stage 13 --use_hier_ctc true --inference_config "+fname+" --inference_nj 200 --test_sets dev.en-de"
                    print(cmd)