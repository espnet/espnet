# ESPnet document generation

## Install related packages

```sh
$ pip install -r requirements.txt
```

## Compile

```sh
$ make html
```

`index.html` will be created at `_build/html/index.html`

## Publish

```sh
# <fork>: your fork repository

# better to make new clone
# (you don't need to do the following everytime)
$ git clone <fork> espnet_doc
$ cd espnet_doc
$ git remote add upstream https://github.com/espnet/espnet.git

# get changes from upstream master
$ git fetch upstream
$ git merge upstream/master

# compile html
$ cd doc
$ make html

# upload html to github.io
$ cd ../
$ git checkout gh-pages
$ cp -r doc/_build/html/* .
$ git add *.html *.js *.inv _sources _static apis
$ git commit 
$ git push <origin> gh-pages

# make pull request!
```
