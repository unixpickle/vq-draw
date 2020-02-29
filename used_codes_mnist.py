"""
Compute a very detailed entropy measure for each
stage of an MNIST encoder.

Output after running for a little while:

    --- from 5120 samples ---
    stage 0: entropy=4.080271 unique=64
    stage 1: entropy=4.123491 unique=64
    stage 2: entropy=4.081288 unique=64
    stage 3: entropy=4.103436 unique=64
    stage 4: entropy=4.094135 unique=64
    stage 5: entropy=4.050476 unique=64
    stage 6: entropy=4.087869 unique=64
    stage 7: entropy=4.122800 unique=64
    stage 8: entropy=4.135551 unique=64
    stage 9: entropy=4.143271 unique=64

So basically, the latent codes are not completely
uniform, but they are pretty close. And no code is
going unused for any stage.
"""

import torch

from train_mnist import MNISTTrainer


def main():
    trainer = MNISTTrainer()
    encodings = []
    for inputs, _ in trainer.train_loader:
        with torch.no_grad():
            batch_enc = trainer.model(inputs)[0]
            encodings.append(batch_enc)
        all_encodings = torch.cat(encodings, dim=0)
        print('--- from %d samples ---' % all_encodings.shape[0])
        for i in range(trainer.model.num_stages):
            codes = all_encodings[:, i]
            counts = torch.tensor([torch.sum(codes == i) for i in range(trainer.model.options)])
            num_unique = len(set(codes.cpu().numpy().flatten()))
            probs = counts.float() / float(codes.shape[0])
            entropy = -torch.sum(torch.log(probs.clamp(1e-8, 1)) * probs)
            print('stage %d: entropy=%f unique=%d' % (i, entropy.item(), num_unique))


if __name__ == '__main__':
    main()
