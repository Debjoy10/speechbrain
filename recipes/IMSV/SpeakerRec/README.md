# Speaker recognition experiments for I-MSV Challenge.
This folder contains scripts for running speaker identification and verification experiments with the I-MSV Dataset.

# Speaker verification using ECAPA-TDNN embeddings
Run the following command to train speaker embeddings using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

`python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml`

After training the speaker embeddings, it is possible to perform speaker verification using cosine similarity, like:

| Split      | Command |
| ----------- | ----------- |
| Validation, En-En comparisons | `python speaker_verification_cosine.py hparams/verification_ecapa_trained.yaml` |
| Test, Crosslingual comparisons   | `python speaker_verification_cosine.py hparams/verification_ecapa_trained_test.yaml` |

**FOR TEST ONLY:** In case of compute issues, break config file into multiple smaller config files containing less comparisons each. Bash file `< BASH_FILE >` creates and runs all files and combines the predictions into a single score.txt
```
python bash/genbash_test.py --yaml YAML  Main Yaml File
                            --bash BASH_FILE  Location of Bash File
bash bash/< BASH_FILE >
```

# To generate the submission file - 
```
python submission.py [-h] [--scores_file CSV FILE GENERATED IN TESTING]
```

## PreTrained Model + Easy-Inference
You can find the pre-trained ECAPA-TDNN model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

