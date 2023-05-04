# Environment

- Environments (obtained by `$ get_sys_info.sh`)
    - date: `Sun Apr 12 19:51:29 UTC 2020`
    - system information: `Linux b98c0a61b0af 4.14.0-115.el7a.ppc64le #1 SMP Tue Sep 25 12:28:39 EDT 2018 ppc64le ppc64le ppc64le GNU/Linux`
    - python version: `Python 3.6.9 :: Anaconda, Inc.
    - espnet version: `espnet 0.5.4`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1`
    - Git hash: `25b82f936701444eeaffd9375f81aaf7f51ea8dc`

# Common config to all the experiments:
```
# network architecture
# encoder related
eunits: 320
eprojs: 320
subsample: "1_2_2_1_1" # skip every n frame from input to nth layers
# decoder related
dlayers: 1
dunits: 300
# attention related
adim: 320
aconv-chans: 10
aconv-filts: 100

# hybrid CTC/attention
mtlalpha: 0.5

# minibatch related
batch-size: 10
maxlen-in: 800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen-out: 150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt: adadelta
epochs: 20
patience: 3

# scheduled sampling option
sampling-probability: 0.1
```


# trans_type=phn

```
etype: vggbgrup
elayers: 1
atype: coverage_location
dtype: gru

|================================================================|
| Sum/Avg|  192   7215 | 82.1   13.5    4.5    3.4   21.4  100.0 |
|================================================================|
|  Mean  |  8.0  300.6 | 82.1   13.4    4.5    3.4   21.4  100.0 |
|  S.D.  |  0.0   13.6 |  4.2    3.2    2.2    1.7    4.0    0.0 |
| Median |  8.0  300.5 | 81.8   13.3    3.9    2.9   21.2  100.0 |
`----------------------------------------------------------------'
```

```
etype: bgrup
elayers: 3
atype: coverage_location
dtype: lstm

|================================================================|
| Sum/Avg|  192   7215 | 82.7   12.7    4.6    2.9   20.2  100.0 |
|================================================================|
|  Mean  |  8.0  300.6 | 82.7   12.7    4.6    2.9   20.2  100.0 |
|  S.D.  |  0.0   13.6 |  3.7    2.6    2.2    1.5    3.8    0.0 |
| Median |  8.0  300.5 | 82.4   12.9    4.6    2.8   21.0  100.0 |
`----------------------------------------------------------------'
```

```
etype: bgrup
elayers: 5
atype: location
dtype: gru

|================================================================|
| Sum/Avg|  192   7215 | 82.5   13.1    4.4    3.0   20.5  100.0 |
|================================================================|
|  Mean  |  8.0  300.6 | 82.6   13.0    4.4    3.1   20.5  100.0 |
|  S.D.  |  0.0   13.6 |  3.6    2.8    1.9    1.7    3.9    0.0 |
| Median |  8.0  300.5 | 82.8   12.9    4.2    2.6   20.0  100.0 |
`----------------------------------------------------------------'
```

# trans_type=char

```
etype: vggbgrup
elayers: 1
atype: location
dtype: lstm

|================================================================|
|  Mean  |  8.0  375.3 | 69.6   18.2   12.2    7.8   38.2  100.0 |
|  S.D.  |  0.0   16.3 |  5.1    2.9    4.1    2.9    4.8    0.0 |
| Median |  8.0  374.5 | 69.5   18.1   11.6    7.7   38.1  100.0 |
`----------------------------------------------------------------'
```

```
etype: vggbgrup
elayers: 3
atype: location2d
dtype: gru

|================================================================|
|  Mean  |  8.0  375.3 | 72.0   16.2   11.8    6.0   34.0  100.0 |
|  S.D.  |  0.0   16.3 |  4.7    3.0    3.9    2.5    4.6    0.0 |
| Median |  8.0  374.5 | 72.8   16.0   10.8    5.4   34.7  100.0 |
`----------------------------------------------------------------'
```

```
etype: bgrup
elayers: 5
atype: coverage_location
dtype: lstm

|================================================================|
|  Mean  |  8.0  375.3 | 72.8   16.6   10.5    6.6   33.7  100.0 |
|  S.D.  |  0.0   16.3 |  4.9    3.5    3.7    2.1    4.8    0.0 |
| Median |  8.0  374.5 | 73.3   16.0   10.3    6.5   34.1  100.0 |
`----------------------------------------------------------------'
```
