import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from src.MATCC import MATCC


class TestConfig:
    model_name = "MATCC_csi800"
    GPU = 0
    universe = 'csi800'
    model_param_path = (
        "./model_params/MATCC/csi800/TEST_MATCC_csi800_seed_11132.pth"
    )
    seed = os.path.basename(model_param_path).split("_")[-1][:-4]
    # seed = 15032
    if "Checkpoint" in os.path.basename(model_param_path).split("_"):
        load_check = True
    else:
        load_check = False

    dataset_dir_path = "./dataset"
    metrics_path = f"./metrics/{universe}/{model_name}_{seed}"
    labels_pred_path = f"./label_pred/{universe}"

    if not os.path.exists(model_param_path):
        raise FileExistsError("params not exits!")

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    if not os.path.exists(labels_pred_path):
        os.makedirs(labels_pred_path)

    # 设置模型
    seq_len = 8
    d_feat = 158
    d_model = 256
    n_head = 4
    dropout = 0.5
    gate_input_start_index = 158
    gate_input_end_index = 221
    device = torch.device(
        f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    # 模型初始化
    model = MATCC(d_model=d_model, d_feat=d_feat, seq_len=seq_len,
                  t_nhead=n_head, S_dropout_rate=dropout).to(device)
    if load_check:
        checkpoint = torch.load(model_param_path, map_location=device)
        model.load_state_dict(checkpoint["model_param"])
    else:
        model.load_state_dict(torch.load(
            model_param_path, map_location=device))


def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index(), dtype=np.float64).groupby(
            "datetime").size().values
        # calculate begin index of each batch
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


def _init_data_loader(data, shuffle=True, drop_last=False):
    sampler = DailyBatchSamplerRandom(data, shuffle)
    data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
    return data_loader


def test():
    universe = TestConfig.universe
    with open(f'{TestConfig.dataset_dir_path}/{universe}/{universe}_dl_test.pkl', 'rb') as f:
    #with open(f'./model_params/MATCC/{universe}/TEST_MATCC_{universe}_seed_11132.pth', 'rb') as f:
        dl_test = pickle.load(f)
    print("Data Loaded.")

    test_loader = _init_data_loader(dl_test, shuffle=False, drop_last=False)

    device = TestConfig.device

    # Model
    model = TestConfig.model
    seed = TestConfig.seed
    model_name = TestConfig.model_name

    preds = []
    ic = []
    ric = []
    labels = []

    print("==" * 10 + f"Now is Testing {model_name}_{seed}" + "==" * 10 + "\n")

    model.eval()
    for data in test_loader:
        data = torch.squeeze(data, dim=0)
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1]
        with torch.no_grad():
            pred = model(feature.float()).detach().cpu().numpy()
        preds.append(pred.ravel())
        labels.append(label.ravel())

        daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
        ic.append(daily_ic)
        ric.append(daily_ric)

    predictions = pd.Series(np.concatenate(
        preds), name="score", index=dl_test.get_index())
    labels = pd.Series(np.concatenate(labels), name="label",
                       index=dl_test.get_index())

    metrics = {
        'IC': np.mean(ic),
        'ICIR': np.mean(ic) / np.std(ic),
        'RIC': np.mean(ric),
        'RICIR': np.mean(ric) / np.std(ric)
    }
    print("\nTest Dataset Metrics performance:{}\n".format(metrics))

    # 保存结果
    with open(os.path.join(TestConfig.metrics_path, f"{model_name}_{seed}_test_result.txt"), "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")

    return predictions, labels, metrics

if __name__ == "__main__":
    predictions, labels, _ = test()
    if not os.path.exists("./label_pred"):
        os.mkdir("./label_pred")
    with open(f"./label_pred/{TestConfig.universe}/{TestConfig.universe}_pred_{TestConfig.seed}.pkl", "wb") as f:
        pickle.dump(predictions, f)
    # print(predictions)
    with open(f"./label_pred/{TestConfig.universe}/{TestConfig.universe}_labels_{TestConfig.seed}.pkl", "wb") as f:
        pickle.dump(labels, f)
