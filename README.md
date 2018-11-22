# Uncetainty-Aware Attention for Reliable Interpretation and Prediction, and DropMax: Adaptive Variational Softmax

<br />

# Uncetainty-Aware Attention for Reliable Interpretation and Prediction
+ Jay Heo(KAIST, Co-author), Hae Beom Lee (KAIST, Co-author), Saehoon Kim (AITRICS), Juho Lee (Univ. Oxford), Kwang Joon Kim(Yonsei University College of Medicine), Eunho Yang (KAIST), and Sung Ju Hwang (KAIST)

<b> Update (November 4, 2018)</b> TensorFlow implementation of [Uncetainty-Aware Attention for Reliable Interpretation and Prediction](https://arxiv.org/pdf/1805.09653.pdf) which introduces uncertainty-aware attention mechanism for time-series data (in Healthcare). We model the attention weights as Gaussian distribution with input dependent noise that the model generates attention with small variance when it is confident about the contribution of the gived features and allocates noisy attentions with large variance to uncertainty features for each input.

## Abstract
<p align="center">
<img width="633" height="391" src="https://github.com/OpenXAIProject/UncertintyAttention_DropMax/blob/master/UA-master/ua_model.png">
    </p>
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
$ git clone https://github.com/OpenXAIProject/UncertintyAttention_DropMax.git
```
This code is written in Python2.7 and requires [TensorFlow 1.3](https://www.tensorflow.org/versions/r1.3/install/install_linux). In addition, you need to go through further procedures to download datasets such as [Physionet Challenge 2012](https://physionet.org/physiobank/database/challenge/2012/) and [MIMIC-III dataset](https://mimic.physionet.org/). You will first need to request access for MIMIC-III after completing the CITI "Data or Specimens Only Research" course. For the convenience, I have provided Physionet datasets for mortality task in a form of numpy arrays that you can directly run the models. 

### Run the model
1. I've provided two different scripts for running UA and UA+ models. 
2. Before running, you can specify the size of 'embed_size', the size of the 'hidden_units' in LSTM cells, and the number of recurrent layers that generate attention alpha and beta in run_UA.py and run_UA_plus.py files.
3. Dropouts rates can be adjusted in model_UA.py model_UA_plus.py files.
4. To train and evaluate the model, run command below.
```bash
$ cd UA-master
$ python run_UA.py 
```

### Results
<p align="center">
<img width="949" height="355" src="https://github.com/OpenXAIProject/UncertintyAttention_DropMax/blob/master/UA-master/ua_interpretation_1.PNG">
    </p>
Visualization of contributions for a selected patient on PhysioNet mortality prediction task. MechVent - Mechanical ventilation, DiasABP - Diastolic arterial blood pressure, HR - Heart rate, Temp - Temperature, SysABP - Systolic arterial blood pressure, FiO2 - Fractional inspired Oxygen, MAP - Meanarterial blood pressure, Urine - Urine output, GCS - Glasgow coma score. The table presents the value of physiological variables at the previous and the current timestep. Dots correspond to sampled attention weights.
<p align="center">
<img width="927" height="195" src="https://github.com/OpenXAIProject/UncertintyAttention_DropMax/blob/master/UA-master/ua_interpretation_2.PNG">
    </p>
Uncertainty over prediction strength on PhysioNet Challenge dataset. For all models, we measured the prediction uncertainty by using MC-dropout with 50 samples.
<p align="center">
<img width="924" height="335" src="https://github.com/OpenXAIProject/UncertintyAttention_DropMax/blob/master/UA-master/ua_interpretation_3.PNG">
    </p>
Experiments on prediction reliability. The line charts show the ratio of incorrect predictions as a function of the ratio of correct predictions for all datasets.
<br />

# DropMax: Adaptive Variationial Softmax
+ Hae Beom Lee (KAIST), Juho Lee (Univ. Oxford), Saehoon Kim (AITRICS), Eunho Yang (KAIST), and Sung Ju Hwang (KAIST)

This is the Tensor-Flow implementation for the paper DropMax: Adaptive Variationial Softmax (NIPS 2018) : https://arxiv.org/abs/1712.07834

## Abstract
<img align="right" width="400" src="https://github.com/OpenXAIProject/UncertintyAttention_DropMax/blob/master/dropmax-master/concept.png">

We propose DropMax, a stochastic version of softmax classifier which at each iteration drops non-target classes according to dropout probabilities adaptively decided for each instance. Specifically, we overlay binary masking variables over class output probabilities, which are input-adaptively learned via variational inference. This stochastic regularization has an effect of building an ensemble classifier out of exponentially many classifiers with different decision boundaries. Moreover, the learning of dropout rates for non-target classes on each instance allows the classifier to focus more on classification against the most confusing classes. We validate our model on multiple public datasets for classification, on which it obtains significantly improved accuracy over the regular softmax classifier and other baselines. Further analysis of the learned dropout probabilities shows that our model indeed selects confusing classes more often when it performs classification.

## Reference

If you found the provided code useful, please cite our work.

```
@inproceedings{lee2018dropmax,
    author    = {Hae Beom Lee and Juho Lee and Saehoon Kim and Eunho Yang and Sung Ju Hwang},
    title     = {DropMax: Adaptive Variationial Softmax},
    booktitle = {NIPS},
    year      = {2018}
}
```

### Run examples
1. Move to dropmax-master folder
2. Modify ```--mnist_path```, in ```run.sh```
3. Specify ```--model``` among ```softmax``` or ```dropmax```, in ```run.sh```
4. Run ```run.sh```

### Results
<p align="center">
<img width="933" height="242" src="https://github.com/OpenXAIProject/UncertintyAttention_DropMax/blob/master/dropmax-master/dropmax_result1.JPG">
    </p>
Examples from CIFAR-100 dataset with top-4 and bottom-2 retain probabilities. Blue and red color denotes the ground truths and base model predictions respectively.
<p align="center">
<img width="970" height="243" src="https://github.com/OpenXAIProject/UncertintyAttention_DropMax/blob/master/dropmax-master/dropmax_result2.JPG">
    </p>
Contour plots of softmax and DropMax with different retain probabilities. For DropMax, we sampled the Bernoulli variables for each data point with fixed probabilities.

### Implementation Issue
Note that LeNet is used as the base network for this code, whereas in the paper we used the network in the Tensorflow-Tutorial (refer to https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5). But the training results and tendencies are similar.

<br />

## License
[Apache License 2.0](https://github.com/OpenXAIProject/tutorials/blob/master/LICENSE "Apache")

# XAI Project 

**These works were supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence(의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : UNIST, Korea Univ., Yonsei Univ., KAIST, AItrics  

+ Web Site : <http://openXai.org>

# Contact
Jay Heo, sflame87@kaist.ac.kr
<br />
Haebeom Lee, haebeom.lee@kaist.ac.kr 
