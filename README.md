# UncertintyAttention_DropMax
Attention prediction model based on uncertainty

# DropMax: Adaptive Variationial Softmax
+ Hae Beom Lee (KAIST), Juho Lee (Univ. Oxford), Saehoon Kim (AITRICS), Eunho Yang (KAIST), and Sung Ju Hwang (KAIST)

This is the Tensor-Flow implementation for the paper DropMax: Adaptive Variationial Softmax (NIPS 2018) : https://arxiv.org/abs/1712.07834

## Abstract
<img align="right" width="400" src="https://github.com/haebeom-lee/dropmax/blob/master/concept.png">
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
1. Modify ```--mnist_path```, in ```run.sh```
2. Specify ```--model``` among ```softmax``` or ```dropmax```, in ```run.sh```
3. Run ```run.sh```

### Implementation Issue
Note that LeNet is used as the base network for this code, whereas in the paper we used the network in the Tensorflow-Tutorial (refer to https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5). But the training results and tendencies are similar.
