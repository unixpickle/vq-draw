import torch
from torchvision import datasets, transforms

from vq_draw.losses import MSELoss
from vq_draw.model import Encoder, CIFARRecurrentRefiner
from vq_draw.train import ImageTrainer

IMG_SIZE = 32


class CIFARTrainer(ImageTrainer):
    def denormalize_image(self, img):
        return img*0.5 + 0.5

    @property
    def default_checkpoint(self):
        return 'cifar_model.pt'

    @property
    def default_stages(self):
        return 100

    @property
    def default_segment(self):
        return None

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
                       refiner=CIFARRecurrentRefiner(self.args.options, self.args.stages),
                       loss_fn=MSELoss())


if __name__ == '__main__':
    CIFARTrainer().main()
