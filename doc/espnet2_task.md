# Task class and data input system for training
## Task class

In ESpnet1, we have too many duplicated python modules. 
One of the big purposes of ESPnet2 is to provide a common interface and 
enable us to focus more on the unique parts of each task.

`Task` class is a common system to build training tools for each task, 
ASR, TTS, LM, etc. inspired by `Fairseq Task` idea. 
To build your task, only you have to do is just inheriting `AbsTask` class:

```python
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.abs_espnet_model import AbsESPnetModel

class NewModel(ESPnetModel):
    def forward(self, input, target):
        (...)
        # loss: The loss of the task. Must be a scalar value.
        # stats: A dict object, used for logging and validation criterion
        # weight: A scalar value that is used for normalization of loss and stats values among each mini-batches.
        #     In many cases, this value should be equal to the mini-batch-size
        return loss, stats, weight

class NewTask(AbsTask):
    @classmethod
    def add_task_arguments(cls, parser):
        parser.add_arguments(...)
        (...)

    @classmethod
    def build_collate_fn(cls, args: argparse.Namespace)
        (...)

    @classmethod
    def build_preprocess_fn(cls, args, train):
        (...)

    @classmethod
    def required_data_names(cls, inference: bool = False):
        (...)

    @classmethod
    def optional_data_names(cls, inference: bool = False):
        (...)

    @classmethod
    def build_model(cls, args):
        return NewModel(...)

if __name__ == "__main__":
    # Start training
    NewTask.main()
```

## Data input system
Espnet2 also provides a command line interface to describe the training corpus.
On the contrary, unlike `fairseq` or training system such as `pytorch-lightining`, 
our `Task` class doesn't have an interface for building the dataset explicitly.
This is because we aim at the task related to speech/text only, 
so we don't need such general system so far.

The following is an example of the command lint arguments:

```bash
python -m espnet2.bin.asr_train \
  --train_data_path_and_name_and_type=/some/path/tr/wav.scp,speech,sound \
  --train_data_path_and_name_and_type=/some/path/tr/token_int,text,text_int \
  --valid_data_path_and_name_and_type=/some/path/dev/wav.scp,speech,sound \
  --valid_data_path_and_name_and_type=/some/path/dev/token_int,text,text_int
```

First of all, our mini-batch is always a `dict` object:

```python
# In training iteration
for batch in iterator:
    # e.g. batch = {"speech": ..., "text": ...}
    # Forward
    model(**batch)
```

Where the `model` is same as the model built by `Task.build_model()`.

You can flexibly construct this mini-batch object 
using `--*_data_path_and_name_and_type`.
`--*_data_path_and_name_and_type` can be repeated as you need and 
each `--*_data_path_and_name_and_type` corresponds to an element in the mini-batch.
Also, keep in mind that **there is no distinction between input and target data**.


The argument of `--train_data_path_and_name_and_type` 
should be given as three values separated by commas, 
like `<file-path>,<key-name>,<file-format>`.

- `key-name` specify the key of dict
- `file-path` is a file/directory path for the data source.
- `file-format` indicates the format of file specified by `file-path`. e.g. `sound`, `kaldi_ark`, or etc.


### `scp` file
You can show the supported file format using `--help` option.

```bash
python -m espnet2.bin.asr_train --help
```

Almost all formats are referred as `scp` file  according to Kaldi-ASR.
`scp` is just a text file which has two columns for each line: 
The first indicates the sample id and the second is some value. 
e.g. file path, transcription, a sequence of numbers.


- format=npy
    ```
    sample_id_a /some/path/a.npy
    sample_id_b /some/path/b.npy
    ```
- format=sound
    ```
    sample_id_a /some/path/a.flac
    sample_id_b /some/path/a.wav
    ```
- format=kaldi_ark
    ```
    sample_id_a /some/path/a.ark:1234
    sample_id_b /some/path/a.ark:5678
    ```
- format=text_int
    ```
    sample_id_a 10 2 4 4
    sample_id_b 3 2 0 1 6 2
    ```
- format=text
    ```
    sample_id_a hello world
    sample_id_b It is rainy today
    ```


### `required_data_names()` and `optional_data_names()`
Though an arbitrary dictionary can be created by this system, 
each task assumes that the specific key is given for a specific purpose. 
e.g. ASR Task requires `speech` and `text` keys and
each value is used for input data and target data respectively. 
See again the methods of `Task` class: 
`required_data_names()` and `optional_data_names()`.


```python
class NewTask(AbsTask):
  @classmethod
  def required_data_names(cls, inference: bool = False):
      if not inference:
          retval = ("input", "target")
      else:
          retval = ("input",)
      return retval

  @classmethod
  def optional_data_names(cls, inference: bool = False):
      retval = ("auxially_feature",)
      return retval
```

`required_data_names()` determines the mandatory data names and `optional_data_names()` gives optional data. It means that the other names are allowed to given by command line arguments.

```bash
# The following is the expected argument
python -m new_task \
  --train_data_path_and_name_and_type=filepath,input,sometype \
  --train_data_path_and_name_and_type=filepath,target,sometype \
  --train_data_path_and_name_and_type=filepath,auxially_feature,sometype
# The following raises an error
python -m new_task \
  --train_data_path_and_name_and_type=filepath,unknown,sometype
```

The intention of this system is just an assertion check, so if feel unnecessary, 
you can turn off this checking with `--allow_variable_data_keys true`.

```bash
# Ignore assertion checking for data names
python -m new_task \
  --train_data_path_and_name_and_type=filepath,unknown_name,sometype \
  --allow_variable_data_keys true
```


## Customize `collcate_fn` for PyTorch data loader
`Task` class has a method to customize `collcate_fn`:

```python
class NewTask(AbsTask):
  @classmethod
  def build_collate_fn(cls, args: argparse.Namespace):
    ...
```

`collcate_fn` is an argument of `torch.utils.data.DataLoader` and 
it can modify the data which is received from data-loader. e.g.:

```python
def collcate_fn(data):
    # data is a list of the return value of Dataset class:
    modified_data = (...touch data)
    return modified_data

from torch.utils.data import DataLoader
data_loader = DataLoader(dataset, collcate_fn=collcate_fn)
for modified_data in data_loader:
    ...
```

The type of argument is determined by the input `dataset` class and 
our dataset is always `espnet2.train.dataset.ESPnetDataset`, 
which the return value is a tuple of sample id and a dict of tensor,

```python
batch = ("sample_id", {"speech": tensor, "text": tensor})
```

Therefore, the type is a list of dict of tensor.

```python
data = [
  ("sample_id", {"speech": tensor, "text": tensor}),
  ("sample_id2", {"speech": tensor, "text": tensor}),
  ...
]
```

The return type of collate_fn is supposed to be a tuple of list and a dict of tensor in espnet2, 
so the collcate_fn for `Task` must transform the data type to it.

```python
for ids, batch in data_loader:
  model(**batch)
```

We provide common collate_fn and this function can support many cases, 
so you might not need to customize it. 
This collate_fn is aware of variable sequence features for seq2seq task:

- The first axis of the sequence tensor from dataset must be length axis: e.g. (Length, Dim), (Length, Dim, Dim2), or (Length, ...)
- It's not necessary to make the lengths of each sample unified and they are stacked with zero-padding.
    - The value of padding can be changed.
        ```python
        from espnet2.train.collate_fn import CommonCollateFn
        @classmethod
        def build_collate_fn(cls, args):
            # float_pad_value is used for float-tensor and int_pad_value is used for int-tensor
            return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
        ```
- Tensors which represent the length of each samples are also appended
    ```python
    batch = {"speech": ..., "speech_lengths": ..., "text": ..., "text_lengths": ...}
    ```
- If the feature is not sequential data, this behavior can be disabled.
    ```bash
    python -m new_task --train_data_path_and_name_and_type=filepath,foo,npy
    ```
    ```python
    @classmethod
    def build_collate_fn(cls, args):
        return CommonCollateFn(not_sequence=["foo"])
    ```
