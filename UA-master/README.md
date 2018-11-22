# Uncetainty-Aware Attention for Reliable Interpretation and Prediction
+ Jay Heo(KAIST, Co-author), Hae Beom Lee (KAIST, Co-author), Saehoon Kim (AITRICS), Juho Lee (Univ. Oxford), Kwang Joon Kim(Yonsei University College of Medicine), Eunho Yang (KAIST), and Sung Ju Hwang (KAIST)

<b> Update (November 4, 2018)</b> TensorFlow implementation of [Uncetainty-Aware Attention for Reliable Interpretation and Prediction](https://arxiv.org/pdf/1805.09653.pdf) which introduces uncertainty-aware attention mechanism for time-series data (in Healthcare). We model the attention weights as Gaussian distribution with input dependent noise that the model generates attention with small variance when it is confident about the contribution of the gived features and allocates noisy attentions with large variance to uncertainty features for each input.

## Abstract
Attention mechanism is effective in both focusing the deep learning models on relevant features and interpreting them. However, attentions may be unreliable since the networks that generate them are often trained in a weakly-supervised manner. To overcome this limitation, we introduce the notion of input-dependent uncertainty to the attention mechanism, such that it generates attention for each feature with varying degrees of noise based on the given input, to learn larger variance on instances it is uncertain about. We learn this Uncertainty-aware Attention (UA) mechanism using variational inference, and validate it on various risk prediction tasks from electronic health records on which our model significantly outperforms existing attention models. The analysis of the learned attentions shows that our model generates attentions that comply with clinicians’ interpretation, and provide richer interpretation via learned variance. Further evaluation of both the accuracy of the uncertainty calibration and the prediction performance with “I don’t know” decision show that UA yields networks with high reliability as well.

## Reference
If you found the provided code useful, please cite our work.

```
@inproceedings{heo2018ua,
    author    = {Jay Heo and Hae Beom Lee and Saehoon Kim and Juho Lee and Kwang Joon Kim and Eunho Yang and Sung Ju Hwang},
    title     = {Uncertainty-Aware Attention for Reliable Interpretation and Prediction},
    booktitle = {NIPS},
    year      = {2018}
              }
```

<br/>

## Getting Started

### Prerequisites

First, clone this repo in same directory.
```bash
$ git clone https://github.com/jayheo/UA
```
This code is written in Python2.7 and requires [TensorFlow 1.3](https://www.tensorflow.org/versions/r1.3/install/install_linux). In addition, you need to go through further procedures to download datasets such as [Physionet Challenge 2012](https://physionet.org/physiobank/database/challenge/2012/) and [MIMIC-III dataset](https://mimic.physionet.org/). You will first need to request access for MIMIC-III after completing the CITI "Data or Specimens Only Research" course. For the convenience, I have provided Physionet datasets for mortality task in a form of numpy arrays that you can directly run the models. 

### Run the model
1. I've provided two different scripts for running UA and UA+ models. 
2. Before running, you can specify the size of 'embed_size', the size of the 'hidden_units' in LSTM cells, and the number of recurrent layers that generate attention alpha and beta in run_UA.py and run_UA_plus.py files.
3. Dropouts rates can be adjusted in model_UA.py model_UA_plus.py files.
4. To train and evaluate the model, run command below.
```bash
$ python run_UA.py 
$ python run_UA_plus.py
```
