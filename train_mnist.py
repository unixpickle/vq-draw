import torch
from torchvision import datasets, transforms

from deep_cloost.losses import MSELoss
from deep_cloost.model import Encoder, MNISTRefiner
from deep_cloost.train import Trainer

IMG_SIZE = 28


class MNISTTrainer(Trainer):
    def denormalize_image(self, img):
        return img*0.3081 + 0.1307

    @property
    def default_checkpoint(self):
        return 'mnist_model.pt'

    @property
    def default_stages(self):
        return 10

    @property
    def shape(self):
        return (1, IMG_SIZE, IMG_SIZE)

    def create_datasets(self):
        # Taken from pytorch MNIST demo.
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.args.batch, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=self.args.batch, shuffle=True, **kwargs)
        return train_loader, test_loader

    def create_model(self):
        return Encoder(shape=self.shape,
                       options=self.args.options,
                       refiner=MNISTRefiner(self.args.options, self.args.stages),
                       loss_fn=MSELoss())


if __name__ == '__main__':
    MNISTTrainer().main()
