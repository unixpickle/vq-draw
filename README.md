# deep-cloost

**deep-cloost** is a revision of my [cluster boosting algorithm](https://github.com/unixpickle/seqtree#more-on-clusterboosting) that uses deep neural networks instead of nearest neighbors. It can be used to find compact, discrete representations of data. For example, deep-cloost can encode the meaningful features of MNIST digits into about 30 bits of data.

Unlike traditional [VQ-VAE](https://arxiv.org/abs/1711.00937), the representations produced by deep-cloost are so compact that it is possible to sample random latent codes and recover realistic-looking samples from the original data distribution.

# How it works

**deep-cloost** uses the notion of "iterative refinement" to encode a sample from a dataset. First, I will describe how encoding/decoding works under this framework. Then, I will describe the training procedure.

## Encoding/Decoding

During encoding, the model keeps track of the target (i.e. the image to be encoded), and the current reconstruction. The current reconstruction always starts out the same, usually as some tensor of 0s. At each stage *i* of encoding, the model proposes *K* variations (i.e. refinements) of the current reconstruction. The variation with the lowest reconstruction error is chosen, and the index of this variation (from 1 to K) is saved as latent component *i*. After *N* stages of encoding, we have latent components *[c<sub>1</sub>, ..., c<sub>N</sub>]*, forming the complete latent code. Thus, the latent code is *N log<sub>2</sub>(K)* bits.

Decoding is very straightforward. All we have to do is start with the initial reconstruction tensor (all 0s), then iteratively feed in the current reconstruction and select the variation at index *c<sub>i</sub>*.

## Training

It should first be noted that the final reconstruction loss is non-differentiable, since we select a specific refinement at every stage of the encoding process. In particular, for an input *R*, our model outputs refinements *[R<sub>1</sub>, ..., R<sub>k</sub>]*, and we select the best refinement *R<sub>i</sub>* and proceed to the next stage of encoding. The other refinements are not used and do not directly contribute to the final reconstruction loss. Furthermore, if refinement *j* is never used (perhaps because it has a bad set of initial parameters), then there will be no gradient signal to improve *R<sub>j</sub>* so that it can be used for encodings later on in training.

To solve the above problems, we use a slightly modified objective. We minimize the chosen *R<sub>i</sub>* at every stage of encoding (directly improving the reconstruction error), but we also minimize a small coefficient times all of the reconstruction errors: *α (R<sub>1</sub> + ... + R<sub>K</sub>)*. Thus, if a refinement option is never being used, it will gradually be pulled closer to the real refinement distribution until it finally is used to encode some sample in the dataset.

To track how well the refinements are distributed, we can look at the entropy of the distribution of latent components. During training, it gradually increases up to *log(K)*, the theoretical limit.
