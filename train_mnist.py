import torch
from torchvision import datasets, transforms

from vq_draw import MSELoss, GaussianLoss, Encoder, MNISTRefiner, ImageTrainer

IMG_SIZE = 28


class MNISTTrainer(ImageTrainer):
    def denormalize_image(self, img):
        return img*0.3081 + 0.1307

    @property
    def supports_gaussian(self):
        return True

    @property
    def default_checkpoint(self):
        return 'mnist_model.pt'

    @property
    def default_stages(self):
        return 10

    @property
    def default_segment(self):
        return None

    @property
    def shape(self):
        if self.args.gaussian:
            return (1, IMG_SIZE, IMG_SIZE, 2)
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
                       refiner=MNISTRefiner(self.args.options, self.args.stages,
                                            gaussian=self.args.gaussian),
                       loss_fn=MSELoss() if not self.args.gaussian else GaussianLoss())


if __name__ == '__main__':
    MNISTTrainer().main()
