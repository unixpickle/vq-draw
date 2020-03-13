from vq_draw import MNISTDistillAE, ImageDistiller

from train_mnist import create_datasets, create_model

IMG_SIZE = 28


class MNISTDistiller(ImageDistiller):
    def denormalize_image(self, img):
        return img*0.3081 + 0.1307

    @property
    def default_vqdraw_checkpoint(self):
        return 'pretrained/mnist_model.pt'

    @property
    def default_checkpoint(self):
        return 'mnist_distill.pt'

    @property
    def default_stages(self):
        return 10

    @property
    def default_segment(self):
        return None

    @property
    def shape(self):
        return (1, IMG_SIZE, IMG_SIZE)

    def create_datasets(self):
        return create_datasets(self.args.batch, self.use_cuda)

    def create_vqdraw_model(self):
        return create_model(self.shape, self.args.stages, self.args.options, False)

    def create_model(self):
        return MNISTDistillAE(self.args.stages, self.args.options)


if __name__ == '__main__':
    MNISTDistiller().main()
