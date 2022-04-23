for split in ["train","dev","fisher_dev", "fisher_dev2", "fisher_test", "callhome_devtest", "callhome_evltest"]:
    src_file=open("data/"+split+"/src_file.txt","w")
    tgt_file=open("data/"+split+"/tgt_file.txt","w")
    for line1 in open("data/"+split+"/text.lc.rm.es"):
        src_file.write(line1.split()[0]+" <tag_es>\n")
        tgt_file.write(line1.split()[0]+" <tag_en>\n")

