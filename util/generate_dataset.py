import pickle
import torch
import pandas as pd
import numpy as np
import pprint as pp
import os
import argparse
import yaml
from qlib.data.dataset.handler import DataHandlerLP
from qlib.tests.data import GetData
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.constant import REG_CN
import qlib
import sys
from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="csi300",
                        help="dataset type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    with open(f"./2023.yaml", 'r') as f:
        config = yaml.safe_load(f)

    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = DIRNAME / f'handler_{config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")}' \
                       f'_{config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")}.pkl'
    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        print('Save preprocessed data to', h_path)
    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"

    print(config)
    print("\n" + "==" * 20 + "\n")

    dataset = init_instance_by_config(config['task']["dataset"])

    dl_test = dataset.prepare(
        "test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    dl_valid = dataset.prepare(
        "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    dl_train = dataset.prepare(
        "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

    if not os.path.exists("./dataset/data/csi300"):
        os.makedirs("./dataset/data/csi300")

    if not os.path.exists("./dataset/data/csi800"):
        os.makedirs("./dataset/data/csi800")

    with open(f"./dataset/data/{args.universe}/{args.universe}_dl_test.pkl", "wb") as f:
        pickle.dump(dl_test, f)

    with open(f"./dataset/data/{args.universe}/{args.universe}_dl_valid.pkl", "wb") as f:
        pickle.dump(dl_valid, f)

    with open(f"./dataset/data/{args.universe}/{args.universe}_dl_train.pkl", "wb") as f:
        pickle.dump(dl_train, f)

    if os.path.exists(f"{h_path}"):
        os.remove(f"{h_path}")
