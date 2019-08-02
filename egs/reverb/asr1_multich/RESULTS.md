# ASR and Speech Enhancement RESULTS

## End-to-end Enhancement 
### Description
 (1) Training: 2-channel simulation data from REVERB and clean data from WSJ (both WSJ0 and WSJ1);\
 (2) Validation: REVERB 8-channel real and simulation development sets;\
 (3) Evaluation: REVERB 8-channel real and simulation evaluation sets.

### Word Error Rate of Evaluation set
#### 8ch - E2E

```
SimData_et_near_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	5904	|	94.7	4.4	0.9	1.1	6.4	44.9	|
SimData_et_far_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	5904	|	94.2	5.0	0.8	0.8	6.7	45.7	|
SimData_et_near_room2:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	6223	|	94.7	4.4	0.9	0.7	6.0	47.7	|
SimData_et_far_room2:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	6223	|	93.8	5.3	0.9	0.9	7.0	52.1	|
SimData_et_near_room3:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	362	5863	|	94.4	4.8	0.8	0.8	6.4	47.2	|
SimData_et_far_room3:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	362	5863	|	92.8	6.4	0.9	1.2	8.5	53.0	|
RealData_et_near_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	186	3131	|	90.0	8.6	1.5	2.3	12.3	68.3	|
RealData_et_far_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	186	2962	|	89.1	9.7	1.3	1.9	12.8	66.1	|
```

### Speech enhancement scores of Evaluation set
#### 8ch - E2E - Score simulation data

```

======================================
           Cepstral distance in dB
--------------------------------------
            	  mean	median
--------------------------------------
            	   enh	   enh
--------------------------------------
  et_far_room1	  3.42	  3.05
  et_far_room2	  4.36	  3.95
  et_far_room3	  4.07	  3.71
 et_near_room1	  3.37	  2.96
 et_near_room2	  4.06	  3.53
 et_near_room3	  3.88	  3.42
--------------------------------------
       average	  3.86	  3.44
======================================


======================================
            SRMR  (only mean used)
--------------------------------------
            	  mean	median
--------------------------------------
            	   enh	   enh
--------------------------------------
  et_far_room1	  8.42	     -	     
  et_far_room2	  7.34	     -	     
  et_far_room3	  6.53	     -	     
 et_near_room1	  7.62	     -	     
 et_near_room2	  6.76	     -	     
 et_near_room3	  7.04	     -	     
--------------------------------------
       average	  7.29	     -
======================================


======================================
             Log likelihood ratio
--------------------------------------
            	  mean	median
--------------------------------------
            	   enh	   enh
--------------------------------------
  et_far_room1	  0.65	  0.63
  et_far_room2	  0.58	  0.54
  et_far_room3	  0.80	  0.77
 et_near_room1	  1.05	  1.05
 et_near_room2	  0.58	  0.54
 et_near_room3	  0.84	  0.81
--------------------------------------
       average	  0.75	  0.72
======================================


======================================
Frequency-weighted segmental SNR in dB
--------------------------------------
            	  mean	median
--------------------------------------
            	   enh	   enh
--------------------------------------
  et_far_room1	  3.91	  4.65
  et_far_room2	  3.75	  6.74
  et_far_room3	  2.98	  5.26
 et_near_room1	  3.67	  4.42
 et_near_room2	  5.09	  8.42
 et_near_room3	  3.79	  6.07
--------------------------------------
       average	  3.87	  5.93
======================================


======================================
            PESQ  (only mean used)
--------------------------------------
            	  mean	median
--------------------------------------
            	   enh	   enh
--------------------------------------
  et_far_room1	  2.52	     -	     
  et_far_room2	  2.65	     -	     
  et_far_room3	  2.41	     -	     
 et_near_room1	  2.41	     -	     
 et_near_room2	  2.82	     -	     
 et_near_room3	  2.66	     -	     
--------------------------------------
       average	  2.58	     -
======================================
```

#### 8ch - E2E - Score real data

```

======================
            SRMR
----------------------
            	   enh
----------------------
  et_far_room1	  7.41
 et_near_room1	  7.28
----------------------
       average	  7.35
======================
```

## Pipeline Enhancement (For comparison)
### Description
In the pipeline enhancement approach, we still use the training data (2ch-REVERB, WSJ0 and WSJ1) described above.
Instead of using the end-to-end fashion to achieve the speech enhancement, conventional pipeline strategy has been
performed. 
WPE and BeamformIt have been employed to enhance the noisy signals. 
The enhanced speeches were then fed into the encoder-decoder based backend.

### Word Error Rate of Evaluation set
#### 8ch - Noisy - No Frontend

```
SimData_et_near_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	5904	|	95.0	4.1	0.8	0.7	5.7	45.5	|
SimData_et_far_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	5904	|	94.3	4.8	0.9	0.7	6.4	47.4	|
SimData_et_near_room2:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	6223	|	93.3	5.6	1.1	0.8	7.5	55.4	|
SimData_et_far_room2:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	6223	|	88.2	9.8	2.0	1.6	13.5	67.8	|
SimData_et_near_room3:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	362	5863	|	92.6	6.3	1.1	1.0	8.3	55.5	|
SimData_et_far_room3:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	362	5863	|	86.1	12.1	1.8	2.2	16.1	73.8	|
RealData_et_near_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	186	3131	|	80.1	17.2	2.7	3.3	23.1	81.2	|
RealData_et_far_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	186	2962	|	79.0	18.1	3.0	3.7	24.8	80.1	|

```

#### 8ch - Pipeline - WPE + BeamformIt

```
SimData_et_near_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	5904	|	94.3	5.0	0.7	0.8	6.5	49.0	|
SimData_et_far_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	5904	|	94.7	4.5	0.7	0.7	6.0	46.6	|
SimData_et_near_room2:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	6223	|	94.8	4.4	0.8	0.5	5.7	44.6	|
SimData_et_far_room2:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	363	6223	|	94.0	5.0	1.0	0.7	6.7	49.6	|
SimData_et_near_room3:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	362	5863	|	93.9	5.3	0.8	0.7	6.8	48.9	|
SimData_et_far_room3:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	362	5863	|	93.2	6.0	0.8	1.0	7.8	53.3	|
RealData_et_near_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	186	3131	|	90.4	8.5	1.1	2.4	12.0	66.1	|
RealData_et_far_room1:
|	SPKR	|	# Snt	# Wrd	|	Corr	Sub	Del	Ins	Err	S.Err	|
|	Sum/Avg	|	186	2962	|	89.6	9.0	1.3	2.4	12.8	59.7	|

```

### Speech enhancement scores of Evaluation set (copied from ../asr1/RESULTS)
#### 8ch - Pipeline - WPE + BeamformIt - Score simulation data

```

==============================================
           Cepstral distance in dB            
----------------------------------------------
            	      	  mean	      	median
----------------------------------------------
            	   org	   enh	   org	   enh
----------------------------------------------
  dt_far_room1	  2.65	  1.97	  2.36	  1.74	
  dt_far_room2	  5.08	  4.66	  4.94	  4.30	
  dt_far_room3	  4.82	  4.03	  4.60	  3.63	
 dt_near_room1	  1.96	  1.67	  1.67	  1.37	
 dt_near_room2	  4.58	  4.33	  4.30	  3.88	
 dt_near_room3	  4.20	  3.71	  3.91	  3.26	
----------------------------------------------
       average	  3.88	  3.39	  3.63	  3.03
==============================================


==============================================
            SRMR  (only mean used)            
----------------------------------------------
            	      	  mean	      	median
----------------------------------------------
            	   org	   enh	   org	   enh
----------------------------------------------
  dt_far_room1	  4.63	  4.91	     -	     -	
  dt_far_room2	  2.94	  5.13	     -	     -	
  dt_far_room3	  2.76	  4.87	     -	     -	
 dt_near_room1	  4.37	  4.62	     -	     -	
 dt_near_room2	  3.67	  4.39	     -	     -	
 dt_near_room3	  3.66	  4.54	     -	     -	
----------------------------------------------
       average	  3.67	  4.74	     -	     -
==============================================


==============================================
             Log likelihood ratio             
----------------------------------------------
            	      	  mean	      	median
----------------------------------------------
            	   org	   enh	   org	   enh
----------------------------------------------
  dt_far_room1	  0.38	  0.33	  0.35	  0.30	
  dt_far_room2	  0.77	  0.56	  0.64	  0.43	
  dt_far_room3	  0.85	  0.52	  0.77	  0.45	
 dt_near_room1	  0.34	  0.34	  0.33	  0.32	
 dt_near_room2	  0.51	  0.50	  0.43	  0.33	
 dt_near_room3	  0.65	  0.50	  0.59	  0.43	
----------------------------------------------
       average	  0.58	  0.46	  0.52	  0.38
==============================================


==============================================
    Frequency-weighted segmental SNR in dB    
----------------------------------------------
            	      	  mean	      	median
----------------------------------------------
            	   org	   enh	   org	   enh
----------------------------------------------
  dt_far_room1	  6.75	  8.99	  8.93	 11.06	
  dt_far_room2	  0.53	  3.84	  0.37	  5.91	
  dt_far_room3	  0.14	  3.76	  0.39	  6.57	
 dt_near_room1	  8.10	  9.50	 10.47	 11.32	
 dt_near_room2	  3.07	  5.10	  4.58	  8.12	
 dt_near_room3	  2.32	  4.54	  4.41	  8.15	
----------------------------------------------
       average	  3.48	  5.96	  4.86	  8.52
==============================================

```

#### 8ch - Pipeline - WPE + BeamformIt - Score real data

```

==============================
            SRMR
------------------------------
            	   org	   enh
------------------------------
  dt_far_room1	  3.51	  6.03	
 dt_near_room1	  4.05	  6.68	
------------------------------
       average	  3.78	  6.36
==============================
```