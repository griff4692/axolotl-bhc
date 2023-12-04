"""
CLI to run inference on a trained model
"""
import transformers
from axolotl.cli import (
    load_cfg,
    print_axolotl_text_art,
)

import os
import sys
import argparse
from tqdm import tqdm
from datasets import load_from_disk
import logging
import numpy as np
np.random.seed(1992)
import ujson
import torch
import pandas as pd
from pathlib import Path

from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from sent_inference_utils import (
    IN_DIR, run_example, load_ent_embeds, load_tools
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = logging.getLogger("axolotl.scripts")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def sent_inference(
    *,
    args,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
    tools,
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

    print('Reading in dataset...')
    visit_meta = {}
    if args.dataset == 'cumc':
        if args.filtered:
            data_dir = f'/nlp/projects/summarization/bhc_data_cleanup/mistral_inference/{args.dataset}_8192'
        else:
            data_dir = f'/nlp/projects/summarization/bhc_data_cleanup/cumc_test'
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)
    elif args.dataset == 'epic':
        if args.filtered:
            data_dir = f'/nlp/projects/summarization/bhc_data_cleanup/mistral_inference/{args.dataset}_8192'
        else:
            data_dir = '/nlp/projects/summarization/bhc_data_cleanup/summarization_deduped_dataset'
        visit_meta = pd.read_csv('/nlp/projects/summarization/bhc_data_cleanup/bhc_test_meta.csv')
        visit_meta = {
            row['visit_id']: row for row in visit_meta.to_dict('records')
        }
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)
        if args.split in data:
            data = data[args.split]
    else:
        if args.filtered:
            data_dir = f'/nlp/projects/summarization/bhc_data_cleanup/mistral_inference/{args.dataset}_8192'
        else:
            data_dir = '/nlp/projects/summarization/bhc_data_cleanup/mimic_test_filt'
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)

    if args.dataset == 'epic' and args.split == 'test':
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

    example_ids = set([x['example_id'] for x in all_ent_probs])
    prev = len(data)
    data = data.filter(lambda row: row['example_id'] in example_ids)
    new = len(data)
    print(f'Entity Probabilities for {new} / {prev} examples. Filtering...')

    span2embed = load_ent_embeds()

    if cfg.landmark_attention:
        from axolotl.monkeypatch.llama_landmark_attn import set_model_mem_id

        set_model_mem_id(model, tokenizer)
        model.set_mem_cache_args(
            max_seq_len=255, mem_freq=50, top_k=5, max_cache_size=None
        )

    model = model.to(cfg.device)
    outputs = []
    for example in tqdm(data):
        out_row = run_example(
            args, cfg, example, out_dir, all_ent_probs, span2embed, tools, model, tokenizer, visit_meta
        )
        outputs.append(out_row)

    df = pd.DataFrame(outputs)
    print(f'Saving predictions to {out_fn}...')
    df.to_csv(out_fn, index=False)

    print(df.select_dtypes(include='number').mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BHC Sent-Level Summarization.')
    parser.add_argument('--data_dir', default='/nlp/projects/summarization/bhc_data_cleanup')

    parser.add_argument('--dataset', default='epic')
    parser.add_argument('--config', default='sent_focus')

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-overwrite', default=False, action='store_true')

    parser.add_argument('--max_examples', default=1000, type=int)
    parser.add_argument('--max_gen_sents', default=50, type=int)
    parser.add_argument('--ent_merge_threshold', default=0.6, type=float)
    parser.add_argument('--max_summary_tokens', default=512, type=int)
    parser.add_argument('--max_prompt_tokens', default=4096, type=int)
    parser.add_argument(
        '--stagant_breaking_pt', default=5, type=int,
        help='Number of times we can generate a sentence that does not cover any new entities before breaking.'
    )

    parser.add_argument('--split', default='test')
    parser.add_argument('-human', default=False, action='store_true')

    # Entity Parameters
    parser.add_argument('--pred_ent_threshold', default=0.81, type=float)

    # Mistral Arguments
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument(
        '--base_model', default='/nlp/projects/summarization/bhc_data_cleanup/mistral_weights/sent_focus_fixed'
    )
    parser.add_argument('--ckpt', default=1500)

    args = parser.parse_args()

    # TODO remove this parameter
    args.filtered = True

    tools = load_tools(args)

    config = Path(os.path.expanduser(f'~/axolotl-bhc/mistral_{args.config}.yml'))

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

    print(f'Starting Sentence-Level Inference...')
    sent_inference(args=args, cfg=parsed_cfg, cli_args=parsed_cli_args, tools=tools)
