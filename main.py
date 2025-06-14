from config.config import *
from train.cross_validation import *
from train.prepare_data import *
import os
import sys
import random
import numpy as np
import torch

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from train.train_model import run_batch_preprocessing



if __name__ == '__main__':
    args, _ = set_config()

    # ✅ 执行并获取预处理数据
    preprocessed_data = run_batch_preprocessing()

    sub_to_run = np.arange(args.subjects)

    # ✅ 把数据传入 PrepareData
    pd = PrepareData(args, external_data=preprocessed_data)
    pd.run(sub_to_run, split=True, expand=True)

    # ✅ 交叉验证
    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.n_fold_CV(subject=sub_to_run, fold=args.fold, reproduce=args.reproduce)


