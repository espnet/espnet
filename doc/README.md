# ESPnet document generation

## Install

We use [travis-sphinx](https://github.com/Syntaf/travis-sphinx) to generate & deploy HTML documentation.

```sh
$ pip install sphinx
$ pip install sphinx_rtd_theme
$ pip install commonmark==0.5.4 recommonmark
$ pip install travis-sphinx
```

## Generate HTML

You can generate local HTML manually using sphinx Makefile

```sh
$ cd <espnet_root>/doc
$ make html
```

or using travis-sphinx

```sh
$ cd <espnet_root>
$ travis-sphinx build --source=doc --nowarn
```

`index.html` will be created at `doc/build/index.html`

## Deploy

When your PR is merged into `master` branch, our [Travis-CI](https://github.com/espnet/espnet/blob/master/.travis.yml) will automatically deploy your sphinx html into https://espnet.github.io/espnet/ by `travis-sphinx deploy`.
