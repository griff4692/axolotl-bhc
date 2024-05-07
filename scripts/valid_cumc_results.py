import os
import pandas as pd

if __name__ == '__main__':
    stats = []
    dataset = 'mimic'
    BS = False

    target_size = 900 if dataset == 'mimic' else 998

    if BS:
        cols = [
            'bs_precision'
        ]
    else:
        cols = [
            'rouge1', 'rouge2', 'ent_recall', 'hallucination_rate', 'num_pred_tokens', 'faithfulness'
        ]

    seen = None
    for pretrained_model in ['mistral', 'zephyr']:
        for experiment in ['baseline_ctd', 'focus', 'frost_esg', 'focus_ablation']:
            for ckpt in ['2500', '3000', '3500', '4000']:
                if BS:
                    fn = os.path.join(
                        f'/nlp/projects/summarization/bhc_data_cleanup/{pretrained_model}_weights',
                        experiment, f'{dataset}_{ckpt}_w_bertscore.csv'
                    )
                else:
                    fn = os.path.join(
                        f'/nlp/projects/summarization/bhc_data_cleanup/{pretrained_model}_weights',
                        experiment, f'{dataset}_{ckpt}_w_metrics.csv'
                    )
                if os.path.exists(fn):
                    df = pd.read_csv(fn)

                    if dataset == 'cumc':
                        invalid_exids = {'2909118_56040054', '6207085_57476017'}
                    else:
                        invalid_exids = {
                            '44908_173941', '30409_173666', '50672_116233', '11520_157511', '25330_137981',
                            '20190_137000', '20843_116002', '3437_131299', '10796_119533', '25314_177523',
                            '17253_172228'
                        }
                    valid = df[~df['example_id'].isin(invalid_exids)]
                    # invalid_exids = df[df['example_id'].isin(invalid_exids)]
                    print(len(valid))

                    if seen is None:
                        seen = set(valid['example_id'])
                    assert len(seen) == len(set(valid['example_id'])) == len(set(valid['example_id']).intersection(seen))

                    for col in cols:
                        if col == 'num_pred_tokens':
                            prev_v = int(df[col].mean())
                            new_v = int(valid[col].mean())
                        else:
                            prev_v = round(df[col].mean(), 3)
                            new_v = round(valid[col].mean(), 3)

                        if prev_v != new_v:
                            print(pretrained_model, experiment, str(ckpt), col, str(prev_v), ' -> ', str(valid[col].mean()))
