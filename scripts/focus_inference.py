"""
CLI to run inference on a trained model
"""
import transformers
from axolotl.cli import (
    load_cfg,
    print_axolotl_text_art,
)

import os
import string
from collections import Counter
import sys
import argparse
import json
from tqdm import tqdm
from datasets import load_from_disk
import logging
import regex as re
import numpy as np
np.random.seed(1992)
import ujson
from transformers import GenerationConfig
import torch
from datetime import datetime
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords

from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from sent_inference_utils import (
    generate_input, load_ent_embeds, get_entity_guidance, load_ent_info, decorate_set_of_ents
)
from utils import INSTRUCTIONS

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = logging.getLogger("axolotl.scripts")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
SAP_BERT = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
IN_DIR = '/nlp/projects/summarization/bhc_data_cleanup'
SPAN_EMBED_DIM = 768
_DEFAULT_PRED_ENT_THRESHOLD = 0.75
_DEFAULT_ENT_MERGE_THRESHOLD = 0.6
INSTRUCTION = 'Play close attention to entities in double brackets {{ }} when generating the next sentence of the BRIEF HOSPITAL COURSE summary.'
PATIENT_TERMS = {'patient', 'pt', 'patient\'s', 'patients', 'patients\''}
BHC_STOPWORDS = set(stopwords.words('english')).union(string.punctuation).union(PATIENT_TERMS)


def run_prompt(cfg, model, tokenizer, prompt):
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.1,
            max_new_tokens=2048,
            min_new_tokens=4,
            # temperature=0.9,
            # top_p=0.95,
            # top_k=40,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        generated = model.generate(
            inputs=batch['input_ids'].to(cfg.device),
            generation_config=generation_config,
        )

    output = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    output = output.replace('<s>', '').replace('</s>', '')
    sep = '### BRIEF HOSPITAL COURSE:'
    assert sep in output
    return output.split(sep)[-1].strip(), output


def split_into_notes(html_str):
    tps = html_str.split('<SEP>')
    notes = []
    curr_note = []
    for tp in tps:
        curr_note.append(tp)
        if tp == '</d>':
            notes.append('<SEP>'.join(curr_note))
            curr_note = []
    return notes


def remove_duplicates_preserve_order(arr):
    """
    Removes duplicates from a list while preserving order.

    :param arr: A list with possible duplicates.
    :return: A new list with duplicates removed, in the same order as the original list.
    """
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def run_example(args, cfg, example, out_dir, all_ent_probs, span2embed, model, tokenizer, visit_meta):
    example_id = example['example_id']
    save_fn = os.path.join(out_dir, f'{example_id}.json')

    if os.path.exists(save_fn) and not args.overwrite:
        print(f'Already exists --> {save_fn}. Skipping...')
        with open(save_fn, 'r') as fd:
            out_row = ujson.load(fd)
            return out_row

    target_no_dup = '\n'.join(remove_duplicates_preserve_order(example['target_sents']))
    notes = split_into_notes(example['source_filt'])

    admit_date = discharge_date = None
    if 'first_date' in example:
        admit_date = datetime.strptime(example['first_date'].split('_')[0], "%m-%d-%y").date()
        discharge_date = datetime.strptime(example['last_date'].split('_')[0], "%m-%d-%y").date()

    source_input = generate_input(
        notes, admit_date=admit_date, discharge_date=discharge_date,
        include_title=len(notes) <= 100
    )

    ent_suffix = '' if args.dataset == 'epic' else f'_{args.dataset}'
    merge_fn = os.path.join(IN_DIR, f'entity_stanza{ent_suffix}_top_ents', f'{example_id}.json')

    # Entity Stuff
    ent_info = load_ent_info(args, example_id, span2embed)
    guidance, ents_in_guidance, pred_source_clusters = get_entity_guidance(
        example_id, all_ent_probs, ent_info['source_ent_clusters'], ent_info['source_ent_types'],
        pred_ent_threshold=args.pred_ent_threshold
    )

    rel_spans = set()
    for k, arr in ents_in_guidance.items():
        for cluster in arr:
            for v in cluster:
                rel_spans.add(v)

    source_transform = decorate_set_of_ents(source_input, rel_spans=rel_spans, add_space=True)

    # Use only 1 word-piece token
    source_transform = re.sub(r'\s?<e>', ' {{', source_transform)
    source_transform = re.sub(r'</e>\s?', '}} ', source_transform)
    source_transform = re.sub(r'\n{2,}', '\n\n', source_transform).strip()
    source_transform = re.sub(r'({{ ){2,}', '{{ ', source_transform)
    source_transform = re.sub(r'(}} ){2,}', '}} ', source_transform)

    instruction = INSTRUCTIONS['sent_planning']
    prompt = f'[INST]\n{instruction}\n\n{source_transform}\n[/INST]\n### BRIEF HOSPITAL COURSE:\n'
    output, full_output = run_prompt(cfg, model, tokenizer, prompt)

    plan_sents = []
    summary_sents = []

    # IFF the output is malformed re-prompt.
    # - repeated entities
    # - no sentence generated

    is_malformed = True
    while is_malformed:
        is_malformed = False
        valid_lines = []
        for line in full_output.split('\n'):
            if line.startswith('### ENTITIES'):
                ents = re.findall(r'{{ ([^}]+) }}', line)
                counter = Counter(ents).most_common()
                if len(counter) > 0 and counter[0][1] >= 3:
                    is_malformed = True
                    print('Re-prompting with unique mentions only')
                    uniq_ents = remove_duplicates_preserve_order(ents)
                    uniq_ent_str = '; '.join(['{{ ' + ent + ' }}' for ent in uniq_ents])
                    sent_num = re.search(r'### ENTITIES (\d+):', line).group(1)
                    valid_lines.append(f'### ENTITIES {sent_num}: {uniq_ent_str}')
                    partial_prompt = '\n'.join(valid_lines).strip() + '\n' + f'### SENTENCE {sent_num}: '
                    output, full_output = run_prompt(cfg, model, tokenizer, partial_prompt)
                    break
                else:
                    valid_lines.append(line)
            else:
                valid_lines.append(line)

    for line in output.split('\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith('### SENTENCE'):
            summary_sents.append(re.sub(r'### SENTENCE \d+:', '', line).strip())
        elif line.startswith('### ENTITIES'):
            plan_sents.append(re.sub(r'### ENTITIES \d+:', '', line).strip())
        else:
            print(f'Malformed output: {line}. Skipping.')

    print('\n\n')
    print(output)
    print('\n\n')

    prediction = '\n'.join(remove_duplicates_preserve_order(summary_sents))
    plan = '\n'.join(plan_sents)

    out_row = {
        'example_id': example['example_id'],
        'reference': target_no_dup,
        'prediction': prediction,
        'plan': plan
    }

    v_meta = visit_meta.get(example.get('visit_id', ''), {})
    out_row.update(v_meta)

    print(f'Saving to {save_fn}')
    with open(save_fn, 'w') as fd:
        json.dump(out_row, fd)

    return out_row


def focus_inference(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
):
    print('Loading model and tokenizer...')
    print(f'cfg.device={cfg.device}')
    model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    default_tokens = {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}
    model = model.to(torch.bfloat16)

    for token, symbol in default_tokens.items():
        # If the token isn't already specified in the config, add it
        if not (cfg.special_tokens and token in cfg.special_tokens):
            tokenizer.add_special_tokens({token: symbol})

    out_dir = os.path.join(cfg.output_dir, f'{args.dataset}_{args.ckpt}')
    out_fn = f'{out_dir}.csv'
    os.makedirs(out_dir, exist_ok=True)

    print('Reading in dataset...')
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

    if args.dataset == 'epic':
        if args.human:
            ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_human.json')
        else:
            ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_test.json')
    elif args.dataset == 'cumc':
        ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_cumc_test.json')
    else:
        assert args.dataset == 'mimic'
        ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_mimic_test.json')

    with open(ent_fn, 'r') as fd:
        all_ent_probs = ujson.load(fd)

    example_ids = set([x['example_id'] for x in all_ent_probs])
    prev = len(data)
    data = data.filter(lambda row: row['example_id'] in example_ids)
    new = len(data)
    print(f'Entity Probabilities for {new} / {prev} examples. Filtering...')

    span2embed = load_ent_embeds()

    model = model.to(cfg.device)
    outputs = []
    for example in tqdm(data):
        out_row = run_example(
            args, cfg, example, out_dir, all_ent_probs, span2embed, model, tokenizer, visit_meta
        )
        outputs.append(out_row)

    df = pd.DataFrame(outputs)
    print(f'Saving predictions to {out_fn}...')
    df.to_csv(out_fn, index=False)

    print(df.select_dtypes(include='number').mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BHC Focus-Plnaning Summarization.')
    parser.add_argument('--data_dir', default='/nlp/projects/summarization/bhc_data_cleanup')

    parser.add_argument('--dataset', default='epic')
    parser.add_argument('--config', default='focus')

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-overwrite', default=False, action='store_true')

    parser.add_argument('--max_examples', default=1000, type=int)

    parser.add_argument('-human', default=False, action='store_true')

    # Mistral Arguments
    parser.add_argument('--pretrained_model', default='zephyr')
    parser.add_argument(
        '--experiment', default='focus'
    )
    parser.add_argument('--ckpt', default=500)

    # Entity Parameters
    parser.add_argument('--pred_ent_threshold', default=0.81, type=float)

    args = parser.parse_args()

    args.base_model = os.path.join(args.data_dir, f'{args.pretrained_model}_weights', args.experiment)
    config = Path(os.path.expanduser(f'~/axolotl-bhc/{args.pretrained_model}_{args.config}.yml'))

    kwargs = {}
    # pylint: disable=duplicate-code
    print(f'Loading config from {config}')
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    parsed_cfg.sample_packing = False
    parsed_cfg.base_model = os.path.join(args.base_model, f'checkpoint-{args.ckpt}')
    assert os.path.exists(parsed_cfg.base_model)
    parsed_cfg.base_model_config = args.base_model
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.inference = True

    print(f'Starting Baseline Inference...')
    focus_inference(cfg=parsed_cfg, cli_args=parsed_cli_args)
