# source: https://gist.github.com/hrwgc/7455343
function validate_url(){
  if [[ `wget -S --spider $1  2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then echo "true"; fi
}

# validate if each file in the list exists
cat downloads/coraal_download_list.txt | while read file_url
do
  file=$(basename $file_url)

  if `validate_url $file_url`; then : ; else echo "does not exist"; fi
done
