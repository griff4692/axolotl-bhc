import numpy as np
from transformers import AutoTokenizer
import pytorch_lightning as pl
import regex as re
from scipy.stats import pearsonr
import torch
from nltk import word_tokenize
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from transformers.optimization import get_linear_schedule_with_warmup


HF_MODEL = '/nlp/projects/summarization/RoBERTa-base-PM-M3-Voc-distill-align-hf'
CKPT_FN = '/nlp/projects/summarization/section_weights/roberta_final/section_rank/rv9i397v/checkpoints/epoch=0-step=88000.ckpt'


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

def _to_str(section):
    curr_str = ''
    if section['raw'].lower() != 'unknown':
        curr_str = section['raw'] + ':\n'
    for sent in section['sents']:
        if len(curr_str) > 0 and curr_str[-1] not in {'\n', '\t', ' '}:
            curr_str += ' '
        curr_str += sent
    return curr_str.strip()


def get_section_logits(sections, model, tokenizer, batch_size=32):
    section_logits = []
    num_chunks = len(sections)
    for i in range(0, num_chunks, batch_size):
        batch = sections[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding='longest',
            pad_to_multiple_of=8,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            section_logits += model.predict_step(inputs)

    return section_logits


def select_sections(note_logits, num_toks, target_tokens=10000):
    priority = list(np.argsort(-np.array(note_logits)))

    sorted_num_toks = [num_toks[i] for i in priority]
    cum_sum = np.cumsum(sorted_num_toks)
    num_under = max(1, len([x for x in cum_sum if x < target_tokens]))
    note_idx_to_add = priority[:num_under]
    note_idx_to_add = list(sorted(note_idx_to_add))

    return note_idx_to_add


def get_attr(tag, attr):
    return re.search(r'\s' + attr + r'=([^ ]+)', tag).group(1).strip('<>: ')


def split_into_sections(html_str):
    tps = html_str.split('<SEP>')
    sections = []
    curr_section_concept = ''
    curr_section_raw = ''
    curr_section_sents = []
    curr_note_tag = ''
    sent_idx_offset = 0
    for tp_idx, tp in enumerate(tps):
        if tp.startswith('<d'):
            curr_note_tag = tp
        if tp.startswith('<h'):
            curr_section_concept = get_attr(tp, 'con')
            curr_section_raw = re.sub(r'[_\s]+', ' ', get_attr(tp, 'raw')).strip()
            curr_section_sents = []
        elif tp.startswith('<s'):
            sent_body = remove_tags_from_sent(tps[tp_idx + 1].strip())
            # sent_body = tps[tp_idx + 1].strip()
            curr_section_sents.append(sent_body)
        elif tp == '</h>':
            n = len(curr_section_sents)
            sections.append({
                'concept': curr_section_concept,
                'note_tag': curr_note_tag,
                'raw': curr_section_raw,
                'sents': curr_section_sents,
                'sent_idxs': list(range(sent_idx_offset, sent_idx_offset + n))
            })
            sent_idx_offset += n
    return sections


def filter_by_section(source_html, filter_model, target_tokens=10000):
    sections = split_into_sections(source_html.replace('<e>', '').replace('</e>', ''))
    header_lens = [len(word_tokenize(x['concept'])) for x in sections]
    section_strs = list(map(_to_str, sections))

    section_logits = get_section_logits(
        section_strs, filter_model['model'], filter_model['tokenizer'], batch_size=16
    )

    section_lens = [a + len(word_tokenize(x)) for a, x in zip(header_lens, section_strs)]
    section_idx_to_add = select_sections(section_logits, section_lens, target_tokens=target_tokens)

    source_filt = filter_sections_from_idx(source_html, section_idx_to_add)
    return source_filt


def load_section_filter(args):
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    model = SectionClassifier.load_from_checkpoint(
        CKPT_FN, tokenizer=tokenizer, hf_name=HF_MODEL, map_location='cpu'
    ).eval().to(args.device)

    return {
        'model': model,
        'tokenizer': tokenizer,
    }
