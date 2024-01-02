import pandas as pd
import os


if __name__ == '__main__':
    exp = 'focus'
    ckpt = '2500'
    pretrained = 'zephyr'
    dataset = 'mimic'

    in_dir = os.path.join('/nlp/projects/summarization/bhc_data_cleanup', f'{pretrained}_weights', exp)
    fn = os.path.join(in_dir, f'{dataset}_{ckpt}_w_metrics.csv')

    if not os.path.exists(fn):
        print(f'Could not find -> {fn}')
        print('Heres what we do have...')
        print([x for x in os.listdir(in_dir) if dataset in x and x.endswith('csv')])
    else:
        print(f'Reading in -> {fn}')
        df = pd.read_csv(fn)
        print(f'Found {len(df)} examples')
        print(df.select_dtypes('number').mean())
