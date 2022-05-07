### description
* add adapters to `wav2vec2` model instances used in [s3prl](https://github.com/s3prl/s3prl)

### how to use
```python3
from add_adapters import add_adapters_wav2vec2

add_adapters_wav2vec2(model, adapter_down_dim=192, adapt_layers=[0, 1, 2])
```

* sample output
```
>> inserted adapters to the following layers: 0, 1, 2
  * original model weights: 95,044,608
  * new model weights - all: 96,819,840
  * new model weights - trainable: 1,775,232 ( 1.87% of original model)
```

* runnable sample code can be found in [example.py](https://github.com/bahducoup/adapter_utils/blob/main/example.py) (`python example.py`)
