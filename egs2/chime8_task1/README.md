# CHiME-8 DASR (CHiME-8 Task 1)

### Distant Automatic Speech Transcription with Multiple Devices in Diverse Scenarios

#### If you want to participate see [official challenge website](https://www.chimechallenge.org/current/task1/index) for registration.


### <a id="reach_us">Any Question/Problem ? Reach us !</a>

If you are considering participating or just want to learn more then please join the <a href="https://groups.google.com/g/chime5/">CHiME Google Group</a>. <br>
We have also a [CHiME Slack Workspace][slack-invite], join the `chime-8-dasr` channel there or contact us directly.<br>

- We also have a [Troubleshooting page](./HELP.md).

### <a id="whatisnew">What is new compared to CHiME-7 DASR Baseline ? </a>

- GSS now is much more memory efficient see https://github.com/desh2608/gss/pull/39 (many thanks to Christoph Boeddeker).
- Segmentation has been retrained on CHiME-6 + NOTSOFAR1 data.
- Some bugs have been fixed.


## System Description



## Results




## Reproducing the Baseline

### Inference-only


### Training the ASR model


### Fine-Tuning the Pyannote Segmentation Model


## Acknowledgements

We would like to thank Naoyuki Kamo for his precious help, Christoph Boeddeker for
reporting many bugs and the memory consumption figures and feedback for evaluation script.


## <a id="reference"> 6. References </a>

[1] Watanabe, S., Mandel, M., Barker, J., Vincent, E., Arora, A., Chang, X., et al. CHiME-6 challenge: Tackling multispeaker speech recognition for unsegmented recordings. <https://arxiv.org/abs/2004.09249> <br>
[2] Van Segbroeck, M., Zaid, A., Kutsenko, K., Huerta, C., Nguyen, T., Luo, X., et al. (2019). DiPCo--Dinner Party Corpus. <https://arxiv.org/abs/1909.13447> <br>
[3] Brandschain, L., Graff, D., Cieri, C., Walker, K., Caruso, C., & Neely, A. (2010, May). Mixer 6. In Proceedings of the Seventh International Conference on Language Resources and Evaluation (LREC'10). <br>
[4] Chang, X., Maekaku, T., Fujita, Y., & Watanabe, S. (2022). End-to-end integration of speech recognition, speech enhancement, and self-supervised learning representation. <https://arxiv.org/abs/2204.00540> <br>
[5] Boeddeker, C., Heitkaemper, J., Schmalenstroeer, J., Drude, L., Heymann, J., & Haeb-Umbach, R. (2018, September). Front-end processing for the CHiME-5 dinner party scenario. In CHiME5 Workshop, Hyderabad, India (Vol. 1). <br>
[6] Kim, S., Hori, T., & Watanabe, S. (2017, March). Joint CTC-attention based end-to-end speech recognition using multi-task learning. Proc. of ICASSP (pp. 4835-4839). IEEE. <br>
[7] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., ... & Wei, F. (2022). Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6), 1505-1518. <br>
[8] Wolf, M., & Nadeu, C. (2014). Channel selection measures for multi-microphone speech recognition. Speech Communication, 57, 170-180.
[9] von Neumann T, Boeddeker C, Kinoshita K, Delcroix M, Haeb-Umbach R. On Word Error Rate Definitions and their Efficient Computation for Multi-Speaker Speech Recognition Systems. arXiv preprint arXiv:2211.16112. 2022 Nov 29.
