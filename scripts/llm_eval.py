import regex as re
import os

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM, GenerationConfig
# Load model directly
import argparse
import itertools
import pandas as pd
from sent_inference_utils import transform_text_for_llama
from nltk import sent_tokenize
from evaluate import load


def run_prompt(prompt, model, tokenizer):
    batch = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)

    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            max_new_tokens=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
        )
        generated = model.generate(
            inputs=batch['input_ids'].to(model.device),
            generation_config=generation_config,
        )

    return generated


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

    parser.add_argument('--dataset', default='epic')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--max_examples', default=1000, type=int)

    parser.add_argument('--split', default='test')
    parser.add_argument('-human', default=False, action='store_true')
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
    model = AutoModelForCausalLM.from_pretrained(
        'HuggingFaceH4/zephyr-7b-beta',
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    ).to(args.device)
    model.eval()

    rouge = load('rouge', keep_in_memory=True)

    args.cached_fn = os.path.join('/nlp/projects/summarization/bhc_data_cleanup/mistral_weights', args.cached_suffix)
    print(f'Loading predictions from from {args.cached_fn}.csv')
    df = pd.read_csv(args.cached_fn + '.csv')
    exid2preds = {row['example_id']: row['prediction'] for row in df.to_dict('records')}

    out_dir = '~/invalid'
    out_fn = args.cached_fn + '_w_llm.csv'
    print(f'Saving metrics to {out_fn}')

    print('Reading in dataset...')
    visit_meta = {}
    if args.dataset == 'cumc':
        data_dir = f'/nlp/projects/summarization/bhc_data_cleanup/cumc_test'
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)
    elif args.dataset == 'epic':
        data_dir = '/nlp/projects/summarization/bhc_data_cleanup/summarization_deduped_dataset'
        visit_meta = pd.read_csv('/nlp/projects/summarization/bhc_data_cleanup/bhc_test_meta.csv')
        visit_meta = {
            row['visit_id']: row for row in visit_meta.to_dict('records')
        }
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)[args.split]
    else:
        data_dir = '/nlp/projects/summarization/bhc_data_cleanup/mimic_test_filt'
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)

    if args.max_examples < len(data):
        data = data.select(range(args.max_examples))

    exid2source = {row['example_id']: row['source'] for row in data}

    outputs = []
    for exid, pred in exid2preds.items():
        out_row = {'example_id': exid}
        source = exid2source[exid]
        assert '<doc-sep>' not in source
        full_source = transform_text_for_llama(
            source, include_header=True, include_title=True, include_sent_markers=False,
            sent_new_line=True
        )

        full_notes = full_source.split('<doc-sep>')
        full_notes = [x.strip() for x in full_notes if len(x.strip()) > 0 and '\n' in x.strip()]
        full_note_sents = [
            [z.strip() for z in x.split('\n') if len(z.strip()) > 0] for x in full_notes
        ]
        full_note_sents_flat = list(itertools.chain(*full_note_sents))

        pred_sents = sent_tokenize_or_parse(pred)

        sent_scores = []
        for pred_sent in pred_sents:
            context, _ = top_k(rouge, pred_sent, full_note_sents_flat)
            system = '<|system|>\nThe information in the SUMMARY sentence can be traced back to the SOURCE.\nDo you agree with this statement?\nAnswer with a single number from 1 (Strongly Disagree) to 5 (Strongly Agree).\n1 - Strongly Disagree\n2 - Disagree\n3 - Neutral\n4 - Agree\n5 - Strongly Agree</s>'
            user = f'<|user|>\nEVIDENCE: {context}\nCLAIM: {pred_sent}</s>'
            assistant = '<|assistant|>\nSCORE: '
            prompt = f'{system}\n{user}\n{assistant}'

            output = run_prompt(prompt, model, tokenizer)

            score = int(output[-1])
            sent_scores.append(score)

        avg_score = np.mean(sent_scores)
        sent_str = ','.join(map(str, sent_scores))
        out_row['sent_scores'] = sent_str
        out_row['score'] = avg_score

        outputs.append(out_row)

        print('Avg Faithfulness: ', round(pd.DataFrame(outputs)['score'].mean(), 3))
