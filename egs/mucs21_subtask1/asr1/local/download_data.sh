path=$1
cwd=`pwd`
DIR=$cwd/$path

declare -A trainset
trainset['Hindi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi_train.tar.gz'
trainset['Marathi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Marathi_train.tar.gz'
trainset['Odia']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Odia_train.tar.gz'

declare -A testset
testset['Hindi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi_test.tar.gz'
testset['Marathi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Marathi_test.tar.gz'
testset['Odia']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Odia_test.tar.gz'

for lang in Hindi Marathi Odia; do
  if [ ! -e ${DIR}/${lang}.done ]; then
      cd ${DIR}
      mkdir -p ${lang}
      cd ${lang}
      wget -O test.zip ${testset[$lang]}
      tar xf "test.zip"
      rm test.zip
      wget -O train.zip ${trainset[$lang]}
      tar xf "train.zip"
      rm train.zip
      cd $cwd
      echo "Successfully finished downloading $lang data."
      touch ${DIR}/${lang}.done
  else
      echo "$lang data already exists. Skip download."
  fi
done
