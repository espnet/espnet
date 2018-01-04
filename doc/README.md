# ESPnet document generation

## Install

```sh
$ pip install sphinx
$ pip install sphinx_rtd_theme
$ pip install commonmark==0.5.4 recommonmark
```

## Compile

```sh
$ make html
```

`index.html` will be created at `_build/html/index.html`

## Publish

```sh
# <origin>: your fork repository
# <upstream>: original repository

# better to make new clone
$ git clone <origin> espnet_doc
$ cd espnet_doc

# get changes from upstream master
$ git fetch <upstream>
$ git merge <upstream>/master

# compile html
$ cd doc
$ make html

# upload html to github.io
$ git checkout gh-pages
$ cp -r doc/_build/html/* .
$ git add *.html *.js *.inv _sources _static apis
$ git commit 
$ git push <origin> gh-pages

# make pull request!
```
