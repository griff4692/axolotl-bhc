import regex as re
import os

import numpy as np
import torch
from datasets import load_from_disk
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sent_inference_utils import transform_text_for_llama
from nltk import sent_tokenize
from evaluate import load
import ujson


def remove_repetitive_sents(pred_sents):
    # Removing repetitive sentences
    pred_sents_tok_cts = [Counter([z.strip() for z in re.split(r'\W+', x) if len(z.strip()) > 0]) for x in pred_sents]
    pred_sents_filt = pred_sents[:1]
    for i in range(1, len(pred_sents)):
        if pred_sents[i].startswith('### SENTENCE') and not pred_sents[i].endswith('.'):
            print(f'Malformed Sentence: {pred_sents[i]}. Removing.')
            continue
        overlap = 0
        total = 0
        for k, v in pred_sents_tok_cts[i].items():
            total += v
            if k in pred_sents_tok_cts[i - 1]:
                overlap += min(v, pred_sents_tok_cts[i - 1][k])
        if total == 0:
            print(f'No tokens in sentence: {pred_sents[i]}')
            continue
        frac = overlap / total
        if overlap < 3 or frac <= 0.5:
            pred_sents_filt.append(pred_sents[i])
        else:
            print('Skipping: ', pred_sents[i])
            print('Because of: ', pred_sents[i - 1])
    return pred_sents_filt


def top_k(rouge, pred, source_sents, k=5):
    scores = [rouge.compute(
        predictions=[s], references=[pred], rouge_types=['rouge1', 'rouge2'],
        use_aggregator=False
    ) for s in source_sents]
    means = [(score_obj['rouge1'][0] + score_obj['rouge2'][0]) / 2.0 for score_obj in scores]
    priority = np.argsort(-np.array(means))
    to_keep = priority[:min(len(priority), k)]
    return '\n'.join([source_sents[i] for i in sorted(to_keep)]), max(means)


def sent_tokenize_or_parse(text):
    has_sent_tags = '<s>' in text
    if has_sent_tags:
        tags = re.split(r'(<\/?s>)', text)
        return [
            tags[i + 1] for i, tag in enumerate(tags) if tag == '<s>'
        ]
    else:
        return sent_tokenize(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LLM Eval Summarization.')
    parser.add_argument('--data_dir', default='/nlp/projects/summarization/bhc_data_cleanup')

    parser.add_argument('--cached_suffix', default='focus/epic_4000')
    parser.add_argument('-summary_level', default=False, action='store_true')

    parser.add_argument('--dataset', default='epic')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--max_examples', default=1000, type=int)

    parser.add_argument('--split', default='test')
    parser.add_argument('-human', default=False, action='store_true')
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    hf = 'allenai/led-large-16384'
    tokenizer = AutoTokenizer.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf).to(args.device).eval()

    args.cached_fn = os.path.join(
        '/nlp/projects/summarization/bhc_data_cleanup/mistral_weights', args.cached_suffix
    )
    print(f'Loading predictions from from {args.cached_fn}.csv')

    df = pd.read_csv(args.cached_fn + '.csv')
    records = df.to_dict('records')
    if args.max_examples < len(df):
        records = records[:args.max_examples]

    exid2preds = {row['example_id']: row['prediction'] for row in records}

    out_dir = args.cached_fn + f'_w_bertscore'
    os.makedirs(out_dir, exist_ok=True)
    full_fn = out_dir + '.csv'
    print(f'Saving metrics to {out_dir}')

    print('Reading in dataset.i..')
    visit_meta = {}
    data_dir = f'/nlp/projects/summarization/bhc_data_cleanup/mistral_inference/{args.dataset}_8192'
    print(f'Reading in data from {data_dir}')
    data = load_from_disk(data_dir)
    if args.dataset == 'epic':
        visit_meta = pd.read_csv('/nlp/projects/summarization/bhc_data_cleanup/bhc_test_meta.csv')
        visit_meta = {
            row['visit_id']: row for row in visit_meta.to_dict('records')
        }

    if args.dataset == 'epic':
        if args.human:
            valid_visit_ids = set(map(str, pd.read_csv(
                '/nlp/projects/summarization/bhc_data_cleanup/bhc_human_meta.csv'
            )['visit_id']))
        else:
            valid_visit_ids = set(map(str, pd.read_csv(
                '/nlp/projects/summarization/bhc_data_cleanup/bhc_test_meta.csv'
            )['visit_id']))
        data = data.filter(
            lambda row: row['visit_id'] in valid_visit_ids
        )
        print(f'{len(data)} examples after removing contaminated samples.')

    n = len(data)
    if args.max_examples is not None and args.max_examples < n:
        idxs = list(sorted(np.random.choice(np.arange(n), size=(args.max_examples), replace=False)))
        data = data.select(idxs)

    print('Building exid2source...')
    exid2source = {row['example_id']: row['source'] for row in data}

    outputs = []
    for exid, pred in exid2preds.items():
        ex_fn = os.path.join(out_dir, f'{exid}.json')
        if os.path.exists(ex_fn) and not args.overwrite:
            print(f'Loading from {ex_fn}...')
            with open(ex_fn, 'r') as fd:
                out_row = ujson.load(fd)
        else:
            out_row = {'example_id': exid}
            source = exid2source[exid]
            assert '<doc-sep>' not in source

            clean_pred = '\n'.join(remove_repetitive_sents(sent_tokenize_or_parse(pred)))

            batch = tokenizer(
                [source, clean_pred], padding='max_length', truncation=True, pad_to_multiple_of=1024, max_length=16384,
                return_tensors='pt'
            )

            global_attention_mask = torch.zeros_like(batch['attention_mask'])
            # put global attention on <s> token
            global_attention_mask[:, 0] = 1

            # since above lists are references, the following line changes the 0 index for all samples
            batch['global_attention_mask'] = global_attention_mask

            batch = {
                k: v.to(model.device) for k, v in batch.items()
            }

            seq_lens = batch['attention_mask'].sum(dim=1)
            with torch.no_grad():
                encodings = model.encoder(**batch).last_hidden_state
                src_h, pred_h = encodings[0, :seq_lens[0], :], encodings[1, :seq_lens[1], :]

                sim_mat = cosine_similarity(pred_h.cpu(), src_h.cpu())

                p = float(sim_mat.max(axis=1).mean())
                r = float(sim_mat.max(axis=0).mean())
                f1 = (2 * p * r) / (p + r)

                out_row['bs_precision'] = p
                out_row['bs_recall'] = r
                out_row['bs_f1'] = f1

            print(f'Saving to {ex_fn}...')
            with open(ex_fn, 'w') as fd:
                ujson.dump(out_row, fd)

        outputs.append(out_row)
        print(f'Avg BERTScore Precision ({len(outputs)}): ', round(pd.DataFrame(outputs)['bs_precision'].mean(), 3))
        print(f'Avg BERTScore Recall ({len(outputs)}): ', round(pd.DataFrame(outputs)['bs_recall'].mean(), 3))
        print(f'Avg BERTScore F1 ({len(outputs)}): ', round(pd.DataFrame(outputs)['bs_f1'].mean(), 3))

    outputs = pd.DataFrame(outputs)
    print(f'Saving faithfulness scores to {full_fn}...')
    print('Final Avg BERTScore: -> ', round(outputs['bertscore'].mean(), 3))
    outputs.to_csv(full_fn, index=False)

