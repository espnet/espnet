#!/bin/bash
rm model_names.txt
python get_model_names.py >> model_names.txt  # Saves model path and model name in a file

IFS=$'\n' GLOBIGNORE='*' command eval  'a=($(cat model_names.txt))'

for value_combine in ${a[@]}
do
echo ${value_combine}
IFS='||' read -ra value <<< ${value_combine}

echo ${value[0]} # This is model path

echo ${value[2]} # This is model name

escapeString="_"
pattern1="/"
pattern2="+" # Repo name does not accept /

repo_name=${value[2]}
repo_name=${repo_name//${pattern1}/${escapeString}} 
repo_name=${repo_name//${pattern2}/${escapeString}} # Get name of hugging face repo

rm -rf dest/*
unzip  ${value[0]} -d dest/ # Save data in dest folder

# # Create hugging face repo and push all data to repo
transformers-cli repo create ${repo_name}  --organization espnet
git clone https://huggingface.co/espnet/${repo_name}
mv dest/* ${repo_name}/.

# Add readme
python create_README_file.py ${repo_name} ${value[2]}
cd ${repo_name}
git add .
git commit -m "import from zenodo"
git push


cd ..
rm -rf ${repo_name}
done
