# Conformer (large model + specaug + TransformerLM)
- Model files
  - training config file: `conf/train.yaml`
  - decoding config file: `conf/decode.yaml`
  - preprocess config file: `conf/specaug.yaml`

- Results
```
Test Set (UnA):
    Ben-Eng:
    | SPKR | # Snt # Wrd | Corr Sub Del Ins Err S.Err |
    | Sum/Avg | 4275 38562 | 66.7 24.4 8.9 3.9 37.2 82.1 |
    Hin-Eng:
    | SPKR | # Snt # Wrd | Corr Sub Del Ins Err S.Err |
    | Sum/Avg | 3136 37611 | 76.6 16.7 6.7 4.3 27.7 62.8 |
```
```
Blind Test Set(UnA):
    Ben-Eng:
    | SPKR | # Snt # Wrd | Corr Sub Del Ins Err S.Err |
    | Sum/Avg | 3130 29157 | 70.1 26.1 3.8 10.6 40.5 87.9 |
    Hin-Eng:
    | SPKR | # Snt # Wrd | Corr Sub Del Ins Err S.Err |
    | Sum/Avg | 4034 44705 | 77.7 17.7 4.6 11.4 33.7 87.0 |
```
