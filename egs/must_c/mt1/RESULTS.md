# Summary (4-gram BLEU)
### Main direction
|model                   |En->De|En->Pt|En->Fr|En->Es|En->Ro|En->Ru|En->Nl|En->It|
|------------------------|-----|-----|-----|-----|-----|-----|-----|-----|
|Transformer (tc->tc) [[Di Gangi et al.]](https://www.aclweb.org/anthology/N19-1202/)|28.09|32.44|42.23|34.16|28.16|18.30|33.43|30.40|
|Transformer (lc.rm->tc) [[Di Gangi et al.]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3045.pdf)|25.30|31.10|35.50|29.90|22.60|14.00|30.30|25.80|
|Transformer (joint BPE8k, tc->tc)    |[30.16](https://drive.google.com/open?id=1vVLRVezCzGhkXwQ9BOWPERmeBPoVRQS_)|[36.65](https://drive.google.com/open?id=1c2pOftpGLXyP67yTTBc4QRa-lSiOCOOY)|[43.02](https://drive.google.com/open?id=1g_mxX9Ql4eshmcEV7Hp2khY3SDpq-nYP)|[35.12](https://drive.google.com/open?id=1C3_o5YYmIqKvrhEOLjKdyMpz23Dy_ngt)|[29.35](https://drive.google.com/open?id=17tQdrrC1roXiDqNEX_j81Lr3XuDk2nL0)|[19.96](https://drive.google.com/open?id=1Itdwh5EJwoP_Wza9dz-CDnee5v98pTKO)|[35.52](https://drive.google.com/open?id=1tBBUqOcAFoTDjteNYXazUDG6wIHkvdN0)|[31.08](https://drive.google.com/open?id=1y124_HW8k16U_oZpYBucinm5l8LZtuT0)|
|Transformer (joint BPE8k, lc.rm->tc) |[27.63](https://drive.google.com/open?id=1qQRu5m99PGR6XW5COgqdYAfYwbSGy39k)|[33.34](https://drive.google.com/open?id=15hpGUyQTLKBLUxcdXdxnxD91f3X1bHWV)|[38.58](https://drive.google.com/open?id=1lBnAbZCSR-y2gz1aWEdt_LxJm7KvBfZJ)|[32.61](https://drive.google.com/open?id=1d9iqY-R0E6DzU1Af9KZQI3gzIuLGLfal)|[25.92](https://drive.google.com/open?id=1x2k-N7DKXYi1WN9uB3qTlwIGxIiuZJNt)|[18.40](https://drive.google.com/open?id=1ZNkmLVR6wlWTU9fmWZc5cLkdlF8LUHwO)|[32.08](https://drive.google.com/open?id=1K881dOzy13UDOr_VzteUm6zETmR_fkgw)|[27.68](https://drive.google.com/open?id=1jnS8aZh-FoKBy1qjX9tJ0wF8qK3weVBY)|

### Reverse direction
|model                   |De->En|Pt->En|Fr->En|Es->En|Ro->En|Ru->En|Nl->En|It->En|
|------------------------|-----|-----|-----|-----|-----|-----|-----|-----|
|Transformer (joint BPE8k, tc->tc)    |35.37|39.64|43.70|35.86|37.97|22.74|38.25|33.39|
