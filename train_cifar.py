import torch
from torchvision import datasets, transforms

from deep_cloost.losses import MSELoss
from deep_cloost.model import Encoder, CIFARRefiner
from deep_cloost.train import Trainer

IMG_SIZE = 32


class CIFARTrainer(Trainer):
    def denormalize_image(self, img):
        return img*0.5 + 0.5

    @property
    def default_checkpoint(self):
        return 'cifar_model.pt'

    @property
    def default_stages(self):
        return 60

    @property
    def shape(self):
        return (3, IMG_SIZE, IMG_SIZE)

    def create_datasets(self):
        # Taken from pytorch MNIST demo.
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./cifar_data', train=True, download=True,
                             transform=transform),
            batch_size=self.args.batch, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./cifar_data', train=False, transform=transform),
            batch_size=self.args.batch, shuffle=True, **kwargs)
        return train_loader, test_loader

    def create_model(self):
        return Encoder(shape=self.shape,
                       options=self.args.options,
                       refiner=CIFARRefiner(self.args.options),
                       loss_fn=MSELoss(),
                       num_stages=self.args.stages)


if __name__ == '__main__':
    CIFARTrainer().main()
