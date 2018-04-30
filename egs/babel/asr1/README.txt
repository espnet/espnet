To setup an experiment (using one or more languages in training) simply run
from this directory ...

./setup_experiment.sh <expname>.

This will copy all the necessary files to the created directory 

../expname

To run the experiment do 

cd ../expname;

To specify the BABEL langauges in training refer to them by their language id.
See  conf/lang.conf for the exhaustive list of languages and corresponding
language ids.

Examples:
./run --langs "102" --recog "102" -- trains on 102 (assamese) and tests on 102.
./run --langs "102 103" --recog " 102 103" -- trains on 102 and 103 (bengali)
                                              and tests on both.

