# VQ-DRAW

**VQ-DRAW** is a [vector quantized](https://en.wikipedia.org/wiki/Vector_quantization) [auto-encoder](https://en.wikipedia.org/wiki/Autoencoder) which encodes inputs in a sequential way similar to the [DRAW](https://arxiv.org/abs/1502.04623) architecture. Unlike [VQ-VAE](https://arxiv.org/abs/1711.00937), it can generate good samples without learning an autoregressive prior on top of the dicrete latents.

In addition to the code for training, I've provided notebooks to play around with some small pre-trained models. These are intended to be runnable on a desktop PC, even without any GPU. Here is a list:

 * [mnist_demo.ipynb](mnist_demo.ipynb) - generate samples and intermediate refinements from an MNIST model
 * [svhn_demo.ipynb](svhn_demo.ipynb) - like mnist_demo.ipynb, but for SVHN.
 * [mnist_classify.ipynb](mnist_classify.ipynb) - train an MNIST classifier on top of features from a pre-trained MNIST encoder.

# Running experiments

```
python -u train_mnist.py --batch 32 --step-limit 50000 --save-interval 500
python -u train_svhn.py --batch 32 --step-interval 16 --step-limit 70000 --save-interval 500
```
