import math

import torch
from torchtext.data import Field, BPTTIterator
from torchtext.datasets import WikiText2

from vq_draw.losses import SoftmaxLoss
from vq_draw.model import Encoder, TextRefiner, SegmentRefiner
from vq_draw.train import TextTrainer


class WikiText2Trainer(TextTrainer):
    def arg_parser(self):
        parser = super().arg_parser()
        parser.add_argument('--bptt-len', default=128, type=int)
        return parser

    @property
    def vocab_size(self):
        return len(self.train_loader.dataset.fields['text'].vocab)

    @property
    def default_checkpoint(self):
        return 'text_model.pt'

    @property
    def default_stages(self):
        return 50

    @property
    def default_segment(self):
        return 10

    @property
    def shape(self):
        return (self.args.bptt_len, self.vocab_size)

    def create_datasets(self):
        field = Field(tokenize=list)
        train, val, test = WikiText2.splits(field, root='wikitext2_data')
        field.build_vocab(train, vectors=None)
        trains, vals, _ = BPTTIterator.splits((train, val, test),
                                              batch_size=self.args.batch,
                                              bptt_len=self.args.bptt_len,
                                              device=torch.device('cpu'))
        return trains, vals

    def create_model(self):
        def make_refiner():
            return TextRefiner(self.args.options,
                               self.args.segment,
                               self.args.bptt_len,
                               self.vocab_size)

        num_refiners = int(math.ceil(self.args.stages / self.args.segment))
        refiner = SegmentRefiner(self.args.segment, *[make_refiner() for _ in range(num_refiners)])
        return Encoder(shape=self.shape,
                       options=self.args.options,
                       refiner=refiner,
                       loss_fn=SoftmaxLoss())

    def cycle_batches(self, loader):
        for x in super().cycle_batches(loader):
            if x.shape[-1] != self.args.bptt_len:
                continue
            yield x


if __name__ == '__main__':
    WikiText2Trainer().main()
