#!/bin/bash
rm model_names.txt
echo "$1"
model_name="$1"
echo ${model_name}

pyscripts/utils/get_model_names.py "${model_name}" >> model_names.txt # Saves model path in a file
declare a
IFS=$'\n' GLOBIGNORE='*' command eval  "a=($(cat model_names.txt))"

for value in "${a[@]}"
do

echo ${value} # This is model path


escapeString="_"
pattern1="/"
pattern2="+"
pattern3=" " # Repo name does not accept /
repo_name=${model_name//${pattern3}/${escapeString}}
repo_name=${repo_name//${pattern1}/${escapeString}}
repo_name=${repo_name//${pattern2}/${escapeString}}
# Get name of hugging face repo

rm -rf dest/*
unzip  ${value} -d dest/ # Save data in dest folder

# # Create hugging face repo and push all data to repo
transformers-cli repo create ${repo_name}  --organization espnet
git clone https://huggingface.co/espnet/${repo_name}
mv dest/* ${repo_name}/.

# Add readme
pyscripts/utils/create_README_file.py ${repo_name} "${model_name}"
cd ${repo_name} || exit
git add .
git commit -m "import from zenodo"
git push


cd ..  || exit
rm -rf ${repo_name}
done
