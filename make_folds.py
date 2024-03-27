# Make folds for cross-validation

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split

INPUT_PATH = r"C:\gatech\bd4h\p1\d_out\input"
LABEL_PATH = r"C:\gatech\bd4h\p1\d_out\label"
FOLD_PATH = r"C:\gatech\bd4h\p1\d_out\input\fold"


def main():
    suffix = "NV"
    data = "mimic"
    files = ['token_type_ids', 'input_ids', 'attention_mask']
    label = "mortality"

    l_data = np.load(rf"{LABEL_PATH}\{data}_{label}.npy", allow_pickle=True)

    # Load data
    # f_data = {}
    # for file_type in files:
    #     f_path = rf"{INPUT_PATH}\{data}\{file_type}_{suffix}.npy"
    #     f_data[file_type] = np.load(f_path, allow_pickle=True)

    length = len(l_data)

    indexes = np.arange(length)

    # Split data
    train_idcs, test_idcs = train_test_split(indexes, test_size=0.2, random_state=2021)
    train_idcs, valid_idcs = train_test_split(train_idcs, test_size=0.25, random_state=2021)

    # convert numpy arrays to dfs
    train_df = pd.DataFrame(data=train_idcs, columns=['index'])
    valid_df = pd.DataFrame(data=valid_idcs, columns=['index'])
    test_df = pd.DataFrame(data=test_idcs, columns=['index'])

    # save dfs to csv mimic_2021_fold_split.csv

    train_df.to_csv(rf"{FOLD_PATH}\{data}_2021_fold_train.csv", index=False)
    valid_df.to_csv(rf"{FOLD_PATH}\{data}_2021_fold_valid.csv", index=False)
    test_df.to_csv(rf"{FOLD_PATH}\{data}_2021_fold_test.csv", index=False)





if __name__ == '__main__':
    SystemExit(main())
