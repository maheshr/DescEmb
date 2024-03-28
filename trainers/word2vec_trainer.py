import os
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.dataset import Word2VecDataset
from models.word2vec import Word2VecModel
from utils.trainer_utils import rename_logger, EarlyStopping

logger = logging.getLogger(__name__)


class Word2VecTrainer():
    def __init__(self, args):
        index_size_dict = {
            # M#: This was nonconcat, renamed to NV
            'nonconcat': {
                'mimic': 1889,
                'eicu': 1534,
                'both': 3223
            },
            'concat_a': {
                'mimic': 70873,
                'eicu': 34424,
                'both': 104353
            },
            'concat_b': {
                'mimic': 70873,
                'eicu': 34424,
                'both': 104353
            },
            'concat_c': {
                'mimic': 70873,
                'eicu': 34424,
                'both': 104353
            },
            'concat_d': {
                'mimic': 3850,
                'eicu': 4354,
                'both': 8095
            },
            'NV': {
                'mimic': 2762,
                # 'eicu': 4354,
                # 'both': 8095
            }
        }

        # M#: Word2VecDataset contructor looks wrong. Rewriting.
        # self.dataloader = DataLoader(dataset=Word2VecDataset(args), batch_size=args.batch_size, shuffle=True)
        self.dataloader = DataLoader(dataset=Word2VecDataset(
            input_path=args.input_path,
            data=args.data,
            eval_data=args.eval_data,
            fold=args.fold,
            split=0.8,
            value_embed_type=args.value_embed_type,
            task=args.task,
            seed=args.seed,
            ratio=args.ratio),
            batch_size=args.batch_size, shuffle=True)

        self.model = Word2VecModel(index_size_dict[args.value_embed_type][args.data], emb_dim=128).cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix

        self.n_epochs = args.n_epochs

        self.early_stopping = EarlyStopping(patience=20, verbose=True)

    def train(self):
        best_loss = float('inf')
        for epoch in range(self.n_epochs):
            avg_loss = 0
            for iter, sample in enumerate(self.dataloader):
                batch_input, batch_labels, batch_neg = sample
                batch_input = batch_input.cuda()
                batch_labels = batch_labels.cuda()
                batch_neg = batch_neg.cuda()

                loss = self.model(batch_input, batch_labels, batch_neg)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()

            avg_loss /= len(self.dataloader)

            logger.info(
                "epoch: {}, loss: {:.3f}".format(
                    epoch, avg_loss
                )
            )

            self.early_stopping(-avg_loss)
            if best_loss > avg_loss:
                best_loss = avg_loss
                logger.info(
                    "Saving checkpoint to {}".format(
                        os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                    )
                )
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict()
                    },
                    os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                )
                logger.info(
                    "Finished saving checkpoint to {}".format(
                        os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                    )
                )

            if self.early_stopping.early_stop is True:
                break
