### Fluent Speech SLU Models


We have implemented the baseline models, and to replicate our baselines, you will need to:
1. Install ESPNet as per the INSTALL file
2. Run the baseline script as 
	`` ./run_baseline.sh --stage -1 --datapath <datapath> --tag <experiment_tag> --ngpu <ngpu> ``

Current Version Result:<br>

**CER result**

| SPKR    | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
|---------|-------|-------|------|-----|-----|-----|-----|-------|
| Sum/Avg | 3793  | 11379 | 98.4 | 1.6 | 0.0 | 0.0 | 1.6 | 3.5   |

**WER result**

| SPKR    | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
|---------|-------|-------|------|-----|-----|-----|-----|-------|
| Sum/Avg | 3793  | 3793  | 96.5 | 3.5 | 0.0 | 0.0 | 3.5 | 3.5   |

Slot ACC, Recall, F1
~~~~ 
                  precision    recall  f1-score   support

           <unk>       0.00      0.00      0.00         0
        ▁Chinese       0.00      0.00      0.00         0
        ▁English       0.91      0.96      0.94        75
         ▁German       0.97      0.98      0.97        57
         ▁Korean       0.96      0.88      0.92        59
       ▁activate       0.92      0.95      0.93        57
        ▁bedroom       0.97      0.97      0.97       663
          ▁bring       0.97      0.97      0.97       392
▁change_language       1.00      0.99      0.99       316
     ▁deactivate       1.00      0.99      0.99       408
       ▁decrease       0.97      0.97      0.97       532
           ▁heat       0.98      0.98      0.98       904
       ▁increase       0.99      0.99      0.99      1133
          ▁juice       0.98      0.99      0.98       970
        ▁kitchen       0.98      0.98      0.98        65
           ▁lamp       1.00      0.99      0.99       423
         ▁lights       0.93      0.99      0.96       144
          ▁music       0.99      0.98      0.99       825
      ▁newspaper       0.99      1.00      0.99       226
           ▁none       1.00      0.96      0.98        79
          ▁shoes       0.99      1.00      0.99      2515
          ▁socks       0.98      0.97      0.97        89
         ▁volume       0.99      0.99      0.99        83
       ▁washroom       0.98      0.98      0.98       741

       micro avg       0.98      0.99      0.98     10756
       macro avg       0.89      0.89      0.89     10756
    weighted avg       0.98      0.99      0.98     10756

accuracy score: 0.9844
accuracy with three tokens (action+object+location):0.9649
~~~~ 
