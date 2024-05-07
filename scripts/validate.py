import os

import pandas as pd
from collections import defaultdict


if __name__ == '__main__':
    outputs = []

    bad_examples = {'epic': defaultdict(int), 'cumc': defaultdict(int), 'mimic': defaultdict(int)}

    for dataset in ['epic', 'mimic', 'cumc']:
        for pretrained_model in ['zephyr', 'mistral']:
            for experiment in ['baseline_ctd', 'frost_esg', 'focus', 'focus_ablation']:
                for ckpt in ['2500', '3000', '3500', '4000']:
                    cached_fn = os.path.join(
                        f'/nlp/projects/summarization/bhc_data_cleanup/{pretrained_model}_weights',
                        experiment, f'{dataset}_{ckpt}'
                    )

                    cached_fn_ext = f'{cached_fn}.csv'
                    if not os.path.exists(cached_fn_ext):
                        continue

                    df = pd.read_csv(cached_fn_ext)
                    bad_ids = df[df['prediction'].isnull()]['example_id'].tolist()
                    prev_n = len(df)
                    df.dropna(subset='prediction', inplace=True)
                    for id in bad_ids:
                        bad_examples[dataset][id] += 1
                    new_n = len(df)
                    null = prev_n - new_n
                    if null > 0:
                        outputs.append({
                            'dataset': dataset,
                            'model': pretrained_model,
                            'experiment': experiment,
                            'ckpt': ckpt,
                            'null': null,
                            'fn': cached_fn_ext
                        })
                        print(outputs[-1])

    print(bad_examples)
