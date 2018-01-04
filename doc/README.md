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
$ cd doc
$ make html
$ git checkout gh-pages
$ cp -r doc/_build/html/* .
$ rm -r doc src 
$ git add --all
$ git commit 
$ git push <remote_name> gh-pages
```
