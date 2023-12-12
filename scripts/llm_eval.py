import regex as re
import os

import numpy as np
import torch
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM, GenerationConfig
# Load model directly
import argparse
import itertools
import pandas as pd
from sent_inference_utils import transform_text_for_llama
from nltk import sent_tokenize
from evaluate import load
import ujson


SYSTEM = 'The information in the SUMMARY can be traced back to the preceeding SOURCE.\nDo you agree with this statement?\nAnswer with a single number from 1 (Strongly Disagree) to 5 (Strongly Agree).\n1 - Strongly Disagree\n2 - Disagree\n3 - Neutral\n4 - Agree\n5 - Strongly Agree'


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


def run_prompt(prompt, model, tokenizer, max_new_tokens=2):
    batch = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)

    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            do_sample=False,
            use_cache=True,
        )

        generated = model.generate(
            inputs=batch['input_ids'].to(model.device),
            generation_config=generation_config,
        )

    output = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    output = output.replace('<s>', '').replace('</s>', '').strip()
    return output


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

    # tokenizer = AutoTokenizer.from_pretrained('TheBloke/Yi-34B-Chat-AWQ')
    # model = AutoModelForCausalLM.from_pretrained(
    #     # 'HuggingFaceH4/zephyr-7b-beta',
    #     'TheBloke/Yi-34B-Chat-AWQ',
    #     # torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage = True,
    #     use_flash_attention_2=True,
    #     device_map=f'cuda:{args.device}'
    # )
    # model.eval()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
    llm = LLM(model='TheBloke/Yi-34B-Chat-AWQ', quantization='awq', dtype='auto', gpu_memory_utilization=0.95)

    rouge = load('rouge', keep_in_memory=True)

    args.cached_fn = os.path.join('/nlp/projects/summarization/bhc_data_cleanup/mistral_weights', args.cached_suffix)
    print(f'Loading predictions from from {args.cached_fn}.csv')

    df = pd.read_csv(args.cached_fn + '.csv')
    records = df.to_dict('records')
    if args.max_examples < len(df):
        records = records[:args.max_examples]

    exid2preds = {row['example_id']: row['prediction'] for row in records}

    granularity = 'summary' if args.summary_level else 'sent'
    out_dir = args.cached_fn + f'_w_{granularity}_faith'
    os.makedirs(out_dir, exist_ok=True)
    full_fn = out_dir + '.csv'
    print(f'Saving metrics to {out_dir}')

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

            # if args.summary_level:
            #     context, _ = top_k(rouge, pred, full_note_sents_flat, k=5)
            #     system = 'The information in the SUMMARY can be traced back to the preceeding SOURCE.\nDo you agree with this statement?\nAnswer with a single number from 1 (Strongly Disagree) to 5 (Strongly Agree).\n1 - Strongly Disagree\n2 - Disagree\n3 - Neutral\n4 - Agree\n5 - Strongly Agree'
            #     user = f'SOURCE:\n{context}\nSUMMARY:\n{pred}'
            #     prompt = f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant'
            #
            #     try:
            #         output = llm.generate(prompt, sampling_params)
            #     except Exception as e:
            #         print(e)
            #         print('Could not run generate on model. Trying with fewer sentences...')
            #         context, _ = top_k(rouge, pred, full_note_sents_flat, k=1)
            #         user = f'SOURCE:\n{context}\nSUMMARY:\n{pred}'
            #         prompt = f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant'
            #     score = int(output[-1])
            #     sent_scores = [int(output[-1])]
            # else:
            pred_sents = sent_tokenize_or_parse(pred)
            pred_sents = remove_repetitive_sents(pred_sents)

            sent_scores = []
            for pred_sent in pred_sents:
                context, _ = top_k(rouge, pred_sent, full_note_sents_flat, k=5)
                user = f'SOURCE:\n{context}\nSUMMARY:\n{pred_sent}'
                prompt = f'<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\nINTEGER ANSWER: '
                try:
                    output = llm.generate(prompt, sampling_params, use_tqdm=False)
                    score = int(output[0].outputs[0].text.strip()[0])
                except Exception as e:
                    print(e)
                    print(f'Could not process. Error is above. Trying with 1-sentence context...')
                    short_context, _ = top_k(rouge, pred_sent, full_note_sents_flat, k=1)
                    short_user = f'SOURCE:\n{short_context}\nSUMMARY:\n{pred_sent}'
                    short_prompt = f'<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{short_user}<|im_end|>\n<|im_start|>assistant\nINTEGER ANSWER: '
                    output = llm.generate(short_prompt, sampling_params, use_tqdm=False)
                    score = int(output[0].outputs[0].text.strip()[0])
                sent_scores.append(score)

            avg_score = np.mean(sent_scores)
            sent_str = ','.join(map(str, sent_scores))
            out_row['sent_scores'] = sent_str
            out_row['score'] = avg_score

            print(f'Saving to {ex_fn}...')
            with open(ex_fn, 'w') as fd:
                ujson.dump(out_row, fd)

        outputs.append(out_row)
        print(f'Avg Faithfulness ({len(outputs)}): ', round(pd.DataFrame(outputs)['score'].mean(), 3))

    outputs = pd.DataFrame(outputs)
    print(f'Saving faithfulness scores to {full_fn}...')
    print('Final Avg Faithfulness: -> ', round(outputs['score'].mean(), 3))
    outputs.to_csv(full_fn, index=False)

