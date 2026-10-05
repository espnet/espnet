# Structured annotations in ESPnet2

The `json` dataset type reads an utterance ID followed by a JSON object:

```text
utt1 {"confidence": 0.75, "count": 3}
utt2 {"count": 2}
```

For example, `--train_data_path_and_name_and_type annotations.scp,annotations,json`
loads these objects under `annotations`. The name is chosen by the task; shared
loading and statistics collection do not reserve a `metrics` field.
`JsonScpReader` validates that each value is an object. `MetricReader` and the
`metric` dataset type remain aliases for historical configurations.

Tasks choose how to interpret and collate annotations. `MappingCollateFn` handles
flat numeric labels with an explicit schema:

```python
collate = MappingCollateFn(
    fields={"annotations": ["confidence", "count"]},
    mapping_pad_value=-100,
)
```

Each declared label becomes a batch of float32 scalars. Missing labels use the
configured padding value; sequence inputs keep normal `CommonCollateFn` padding
and lengths. Categorical strings and nested objects require task-specific
preprocessing/collation. They are not implicitly converted into numeric labels.
`UniversaCollateFn` wraps this mechanism for the historical `metrics` field.
The numeric `UniversaProcessor` rejects string metric types; ARECHO uses a
separate tokenizing preprocessor for its categorical labels.

Statistics collection skips mapping-valued annotations by runtime type, under
any field name. A tensor named `metrics` still receives a shape file and an entry
in `batch_keys`.

## Missing audio

A `None` entry in a sound SCP represents an unavailable waveform. Dataset
preprocessing must resolve it before collation. Both indexed and streaming
loaders reject unresolved missing values. Literal `None` in a text input is text.

`UniversaProcessor` replaces only its configured `ref_audio_name` with silence.
Its required `audio_name` raises an error when missing; unrelated missing inputs
remain errors at the dataset boundary. This policy is opt-in through that
preprocessor and does not silently substitute silence in existing recipes.
