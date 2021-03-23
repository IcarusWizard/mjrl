import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    metrics = pd.read_csv(args.path)
    performance = metrics['eval_score'].to_numpy()[-101:-1].mean()

    print(performance)
