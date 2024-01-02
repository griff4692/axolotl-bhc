import os

import argparse
import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LLM Eval Summarization.')
    parser.add_argument('--data_dir', default='/nlp/projects/summarization/bhc_data_cleanup')
    args = parser.parse_args()

    stats = []

    for pretrained_model in ['mistral', 'zephyr']:
        for experiment in ['baseline_ctd', 'focus', 'frost_esg']:
            for dataset in ['epic', 'mimic', 'cumc']:
                print(f'Starting dataset {dataset}')
                for ckpt in ['2500', '3000', '3500', '4000']:
                    print(f'Starting checkpoint {ckpt}')
                    fn = os.path.join(
                        f'/nlp/projects/summarization/bhc_data_cleanup/{pretrained_model}_weights',
                        experiment, f'{dataset}_{ckpt}_w_bertscore.csv'
                    )

                    row = {
                        'model': pretrained_model,
                        'dataset': dataset,
                        'experiment': experiment,
                        'ckpt': ckpt,
                    }

                    if os.path.exists(fn):
                        row['bs_precision'] = pd.read_csv(fn)['bs_precision'].mean()

                    stats.append(row)

    stats = pd.DataFrame(stats)

    for record in stats.to_dict('records'):
        if np.isnan(record['bs_precision']):
            continue
        print(record)
