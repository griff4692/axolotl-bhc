import os

from transformers import AutoTokenizer

HF_MODEL = '/nlp/projects/summarization/RoBERTa-base-PM-M3-Voc-distill-align-hf'
CKPT_FN = '/nlp/projects/summarization/section_weights/roberta_final/section_rank/rv9i397v/checkpoints/epoch=0-step=88000.ckpt'

import pytorch_lightning as pl
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from transformers.optimization import get_linear_schedule_with_warmup


class SectionClassifier(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_name):
        """
        :param args:
        :param tokenizer:
        :param hf_name:
        """
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        print(f'Loading {hf_name}')

        self.config = RobertaConfig.from_pretrained(hf_name)
        self.model = RobertaModel.from_pretrained(hf_name, config=self.config, add_pooling_layer=False)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.loss = torch.nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        self.model.resize_token_embeddings(len(tokenizer))

    def compute_corel(self, y, y_pred):
        y_pred_prob_np = torch.sigmoid(y_pred.detach()).cpu().numpy()
        y_np = y.cpu().numpy()
        try:
            corel = pearsonr(y_pred_prob_np, y_np)[0]
        except:  # correlation is undefined if one array is constant or both are of length 1
            print(y_pred_prob_np, y_np)
            print('oops')
            corel = None
        return corel

    def shared_step(self, batch, labels):
        h = self.dropout(self.model(**batch).last_hidden_state[:, 0, :])  # Effectively take CLS token
        logits = self.classifier(h).squeeze(-1)
        loss = self.loss(logits, torch.clamp(labels * self.hparams.label_multiplier, min=0.0, max=1.0))
        corel = self.compute_corel(labels, logits)
        return loss, corel

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels', None)

        loss, corel = self.shared_step(batch, labels)

        if corel is not None:
            self.log('train/corel', corel, on_epoch=False, on_step=True, prog_bar=True, sync_dist=True)

        self.log('train/loss', loss, on_epoch=False, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels', None)

        loss, corel = self.shared_logistic_step(batch, labels)

        if corel is not None:
            self.log('val/corel', corel, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch):
        with torch.no_grad():
            h = self.dropout(self.model(**batch).last_hidden_state[:, 0, :])  # Effectively take CLS token
            logits = torch.sigmoid(self.classifier(h).squeeze(-1))
            return logits.detach().cpu().numpy().tolist()

    def configure_optimizers(self):
        nps = list(self.named_parameters())
        grouped_parameters = [
            {
                'params': [p for n, p in nps if 'classifier' in n and p.requires_grad],
                'lr': 1e-3,
            },
            {
                'params': [p for n, p in nps if 'classifier' not in n and p.requires_grad],
                'lr': self.hparams.lr
            },
        ]

        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.hparams.lr)
        warmup = min(self.hparams.warmup, self.hparams.max_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=self.hparams.max_steps
        )

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]


def load_section_filter(args):
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    model = SectionClassifier.load_from_checkpoint(
        CKPT_FN, tokenizer=tokenizer, hf_name=HF_MODEL, map_location='cpu'
    ).eval().to(args.device)

    return {
        'model': model,
        'tokenizer': tokenizer,
    }
