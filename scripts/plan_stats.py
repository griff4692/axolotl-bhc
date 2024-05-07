import pandas as pd
import os
import numpy as np


if __name__ == '__main__':
    exp = 'focus_ablation'
    pretrained = 'zephyr'
    dataset = 'epic'

    avgs = {'recall': [], 'precision': [], 'f1': []}
    for ckpt in ['2500', '3000', '3500', '4000']:
        in_dir = os.path.join('/nlp/projects/summarization/bhc_data_cleanup', f'{pretrained}_weights', exp)
        fn = os.path.join(in_dir, f'{dataset}_{ckpt}_w_metrics.csv')

        print(f'Reading in -> {fn}')
        df = pd.read_csv(fn)
        pr = df['plan_recall'].mean()
        pp = df['plan_precision'].mean()
        pf = df['plan_f1'].mean()

        avgs['recall'].append(pr)
        avgs['precision'].append(pp)
        avgs['f1'].append(pf)

    cols = ['recall', 'precision', 'f1']
    vals = [str(round(np.mean(avgs[col]), 3)) for col in cols]
    print(' '.join(cols))
    print(' '.join(vals))
