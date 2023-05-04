# RESULTS
## Environments
- date: `Mon Aug  5 22:20:59 EDT 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.4.1`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `32d38b978f3627fb81602b13e9a81339757085a7`
  - Commit date: `Fri Aug 2 01:53:56 2019 -0400`

## 1. End-to-end Speech Enhancement Results
### 1.1 Description
 (1) Training: 2-channel simulation data from REVERB and clean data from WSJ (both WSJ0 and WSJ1);\
 (2) Validation: REVERB 8-channel real and simulation development sets;\
 (3) Evaluation: REVERB 8-channel real and simulation evaluation sets.

### 1.2 WER
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

### 1.3 Speech Enhancement Scores
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

## 2. Pipeline Enhancement Results (For comparison)
### 2.1 Description
In the pipeline enhancement approach, we still use the training data (2ch-REVERB, WSJ0 and WSJ1) described above.
Instead of using the end-to-end fashion to achieve the speech enhancement, conventional pipeline strategy has been
performed.
WPE and BeamformIt have been employed to enhance the noisy signals.
The enhanced speeches were then fed into the encoder-decoder based backend.

### 2.2 WER
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

### 2.3 Speech enhancement scores
#### 8ch - Pipeline - WPE + BeamformIt - Score simulation data

```

==============================================
           Cepstral distance in dB
----------------------------------------------
            	      	  mean	      	median
----------------------------------------------
            	   org	   enh	   org	   enh
----------------------------------------------
  et_far_room1	  2.67	  2.10	  2.38	  1.84
  et_far_room2	  5.21	  4.94	  5.04	  4.56
  et_far_room3	  4.96	  4.49	  4.73	  4.10
 et_near_room1	  1.99	  1.72	  1.68	  1.39
 et_near_room2	  4.63	  4.43	  4.24	  3.90
 et_near_room3	  4.38	  4.00	  4.04	  3.54
----------------------------------------------
       average	  3.97	  3.61	  3.69	  3.22
==============================================


==============================================
            SRMR  (only mean used)
----------------------------------------------
            	      	  mean	      	median
----------------------------------------------
            	   org	   enh	   org	   enh
----------------------------------------------
  et_far_room1	  4.58	  5.19	     -	     -
  et_far_room2	  2.97	  4.88	     -	     -
  et_far_room3	  2.73	  4.13	     -	     -
 et_near_room1	  4.50	  4.73	     -	     -
 et_near_room2	  3.74	  4.39	     -	     -
 et_near_room3	  3.57	  4.45	     -	     -
----------------------------------------------
       average	  3.68	  4.63	     -	     -
==============================================


==============================================
             Log likelihood ratio
----------------------------------------------
            	      	  mean	      	median
----------------------------------------------
            	   org	   enh	   org	   enh
----------------------------------------------
  et_far_room1	  0.38	  0.33	  0.35	  0.30
  et_far_room2	  0.75	  0.60	  0.63	  0.48
  et_far_room3	  0.84	  0.64	  0.76	  0.57
 et_near_room1	  0.35	  0.34	  0.33	  0.32
 et_near_room2	  0.49	  0.48	  0.40	  0.32
 et_near_room3	  0.65	  0.54	  0.59	  0.46
----------------------------------------------
       average	  0.58	  0.49	  0.51	  0.41
==============================================


==============================================
    Frequency-weighted segmental SNR in dB
----------------------------------------------
            	      	  mean	      	median
----------------------------------------------
            	   org	   enh	   org	   enh
----------------------------------------------
  et_far_room1	  6.68	  8.85	  9.24	 11.33
  et_far_room2	  1.04	  3.17	  1.77	  5.29
  et_far_room3	  0.24	  2.03	  0.89	  3.77
 et_near_room1	  8.12	  9.33	 10.72	 11.48
 et_near_room2	  3.35	  4.75	  5.52	  7.93
 et_near_room3	  2.27	  3.87	  4.21	  6.86
----------------------------------------------
       average	  3.62	  5.33	  5.39	  7.78
==============================================
```

#### 8ch - Pipeline - WPE + BeamformIt - Score real data

```

==============================
            SRMR
------------------------------
            	   org	   enh
------------------------------
  et_far_room1	  3.19	  4.47
 et_near_room1	  3.17	  4.16
------------------------------
       average	  3.18	  4.31
==============================
```
