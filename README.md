# VQ-DRAW

**VQ-DRAW** is a [vector quantized](https://en.wikipedia.org/wiki/Vector_quantization) [auto-encoder](https://en.wikipedia.org/wiki/Autoencoder) which encodes inputs in a sequential way similar to the [DRAW](https://arxiv.org/abs/1502.04623) architecture. Unlike [VQ-VAE](https://arxiv.org/abs/1711.00937), it can generate good samples without learning an autoregressive prior on top of the dicrete latents.

In addition to this write-up and the code, I've provided notebooks to play around with some small pre-trained models. These are [mnist_demo.ipynb](mnist_demo.ipynb) and [svhn_demo.ipynb](svhn_demo.ipynb). They are intended to be easily runnable on a desktop PC, even without any GPU.

# Related work

VQ-DRAW started out as a "deep" variant of my earlier [cluster boosting algorithm](https://github.com/unixpickle/seqtree#more-on-clusterboosting). Cluster boosting is a way of encoding vectors (e.g. images) as a sequence of integers, where each integer indexes a vector and all of the resulting vectors are summed together. Cluster boosting has the limitation that the vectors corresponding to the latent integers are fixed; they do not depend on previous integers in the sequence. Thus, the latent codes have some amount of redundancy that could be avoided. This results in poor samples and relatively large latent codes compared to VQ-DRAW.

Other work has attempted to train discrete VAEs. The [VQ-VAE paper](https://arxiv.org/abs/1711.00937) cites many of these works, and presents a new framework for training discrete VAEs with vector quantization. However, VQ-VAE models only tend to encode local "patches" into each latent integer. This results in spatial grids or temporal sequences of latent integers where individual integers only encode local information. Random sampling of these latent codes results in outputs which are locally coherent but globally incoherent. This is why both VQ-VAE and [VQ-VAE-2](https://arxiv.org/abs/1906.00446) require autoregressive priors on top of the discrete latent representations.

Another line of related work involves the [DRAW](https://arxiv.org/abs/1502.04623) architecture. DRAW works by sequentially generating an image using attention to modify part of the image at a time. Each timestep of the sequence has its own latent code which specifies what local modification to perform. The resulting sequence of latents is treated as one larger latent code for the entire image. While DRAW itself uses attention, follow-up work like [Convolutional DRAW](https://arxiv.org/abs/1604.08772) maintains the same sequential architecture but opts for an attention-free approach. The sequential aspect of VQ-DRAW is similar to DRAW, but the latent codes in VQ-DRAW are discrete and very small, and the neural network architectures used in VQ-DRAW are not recurrent.

# The VQ-DRAW algorithm

This section first describes the encoding/decoding process for a fully trained model. It then presents an algorithm for training such models in practice.

It should be noted that, unlike in DRAW, VQ-DRAW does not have a separate encoder and decoder network. A single network, sometimes referred to as the "encoder" or "model", does the heavy lifting of both encoding and decoding.

## Encoding/Decoding

Both the encoding and decoding processes are sequential. Each timestep of these processes is referred to as a "stage". For ease of notation, the variable *N* refers to the total number of encoding/decoding stages.

The encoding process takes in a tensor and returns a latent code of integers, *[c<sub>1</sub>, ..., c<sub>N</sub>]*. The decoding process approximately reverses this transformation, taking in a latent code of integers and producing a reconstruction tensor.

During encoding, the algorithm keeps track of the target (i.e. the tensor to be encoded), and the current reconstruction. At the first stage, the current reconstruction always starts out the same, usually as some tensor of 0s. At each stage *i* of encoding, the model proposes *K* variations (i.e. refinements) of the current reconstruction. The variation with the lowest reconstruction error is chosen, and the index of this variation (from 1 to K) is saved as latent component *i*. After *N* stages of encoding, we have latent components *[c<sub>1</sub>, ..., c<sub>N</sub>]*, forming the complete latent code. Thus, the latent code is *N log<sub>2</sub>(K)* bits.

Decoding proceeds in a similar fashion. The inputs to this algorithm are the latent components *[c<sub>1</sub>, ..., c<sub>N</sub>]*, and the output is a reconstruction tensor. The algorithm starts with the initial reconstruction tensor (all 0s). For decoding step *i=1* to *N*, it feeds the current reconstruction into the model, selects the variation at index *c<sub>i</sub>*, and sets this variation as the new current reconstruction. After all *N* stages are performed, the current reconstruction tensor is returned.

It should be noted that the encoding process keeps track of the current reconstruction, and therefore gives us the reconstruction for free (i.e. with no extra compute). Thus, during training, no extra decoding step has to be performed.

## Training

First note that it is possible to backpropagate through the encoding process, assuming that the choices for each latent component remain fixed. In other words, the final reconstruction error is locally differentiable, and we can use SGD to minimize it. However, the gradient for a given data point will ignore most of the outputs of the model, since a specific refinement is selected at each stage of the encoding process and the remaining refinements are ignored.

In particular, at stage *i*, our model outputs refinements *[R<sub>i,1</sub>, ..., R<sub>i,k</sub>]*, and we select the best refinement and proceed to the next stage of encoding. The other refinements are not used and do not directly contribute to the final reconstruction loss. Furthermore, if refinement *R<sub>i,j</sub>* is never used for any sample in the dataset (perhaps because it has a bad set of initial biases), then there will be no gradient signal to improve *R<sub>i,j</sub>* so that it can be used for encodings later on in training.

To solve the above problems, we can use a slightly modified objective. First, we directly improve reconstruction error by optimizing the chosen refinement at every stage of encoding. Second, we also minimize a small coefficient times all of the reconstruction errors: *α (R<sub>1,1</sub> + ... + R<sub>N,K</sub>)*. Thus, if a refinement option *R<sub>i,j</sub>* is never being used, it will gradually be pulled closer to the real refinement distribution until it finally is used to encode some sample in the dataset.

To track how well the refinements are distributed, we can look at the entropy of the distribution of latent components for single mini-batches. This metric is not perfect, but it does reveal when a model is not using many of its refinement options effectively. I have found that, with the objective function described above, the entropy typically rises very close to the theoretical limit of *log(K)* (although it cannot reach this limit with a finite batch size).
