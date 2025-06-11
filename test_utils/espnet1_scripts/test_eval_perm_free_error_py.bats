#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import numpy as np
import sys


np.random.seed(0)

def pit_score(samples, num_spkrs=2):
    ret = []
    for idx, samp in enumerate(samples):
        samp_stats = samp['stats']
        scores = [(sum(x[1:]), sum(x[:3])) for x in samp_stats]

        assert num_spkrs == 2
        # the following part should be modified for 3 speakers and more
        wers = [float(scores[0][0]+scores[3][0]) / (scores[0][1] + scores[3][1]),
                float(scores[1][0]+scores[2][0]) / (scores[1][1] + scores[2][1])]

        if wers[0] <= wers[1]:
            ret.append([samp_stats[0], samp_stats[3]])
        else:
            ret.append([samp_stats[1], samp_stats[2]])
    return ret


# generate stats
def generate_stats_sample():
    # C, S, D, I
    stats = []
    for i in range(4):
        stats.append(np.random.randint(0, 20))
    return stats

def generate_samples(num_spkrs=2):
    # generate maximum 10 examples
    nsamples = 2 #np.random.randint(0, 10)
    ret = []
    for i in range(nsamples):
        stats = [generate_stats_sample() for _ in range(num_spkrs ** 2)]
        id = str(i) * 10
        ret.append(dict(id=id, stats=stats))

    return ret

def output_format(id, stats, fh):
    fh.write('Speaker sentences ' + str(int(id[0])+1) + ':\t' + id + '\t#utts: 1\n')
    fh.write('id: (' + id + ')\n')
    new_stats = [str(x) for x in stats]
    fh.write('Scores: (#C #S #D #I) ' + ' '.join(new_stats) + '\n')
    fh.write('REF: this is a random ***** sample\n')
    fh.write('HYP: this is a random WRONG sample\n')
    fh.write('Eval:                 I           \n')
    fh.write('\n')

def main(dir):
    num_spkrs = 2  # Only 2 speakers are supported in scoring
    samples = generate_samples(num_spkrs=num_spkrs)
    for i in range(1, num_spkrs+1):
        for j in range(1, num_spkrs+1):
            k = (i - 1) * num_spkrs + j - 1
            with open(dir+'/result_r{}h{}.wrd.txt'.format(i, j), 'w') as f:
                for idx, samp in enumerate(samples):
                    output_format(samp['id'], samp['stats'][k], f)

    results = pit_score(samples, num_spkrs=num_spkrs)
    Cs = 0
    Ss = 0
    Ds = 0
    Is = 0
    for i, result in enumerate(results):
        Cs += result[0][0] + result[1][0]
        Ss += result[0][1] + result[1][1]
        Ds += result[0][2] + result[1][2]
        Is += result[0][3] + result[1][3]
    with open(dir+'/result_wer.txt', 'w') as f:
        f.write('Total Scores: (#C #S #D #I) {} {} {} {}\n'.format(Cs, Ss, Ds, Is))
        f.write('Error Rate:   {:.2f}\n'.format(float(Ss + Ds + Is) / (Cs + Ss + Ds) * 100))
        f.write('Total Utts:  {}\n'.format(len(results)))

if __name__ == "__main__":
    tmp_dir = '${tmpdir}'
    main(tmp_dir)
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "eval_perm_free_error.sh" {
    python $utils/eval_perm_free_error.py --num-spkrs 2 ${tmpdir}/result_r1h1.wrd.txt \
      ${tmpdir}/result_r1h2.wrd.txt ${tmpdir}/result_r2h1.wrd.txt ${tmpdir}/result_r2h2.wrd.txt | \
      sed -n '2,4p' > ${tmpdir}/min_perm_result.wrd.txt
    diff ${tmpdir}/min_perm_result.wrd.txt ${tmpdir}/result_wer.txt
}
