import copy
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter

from src.MATCC import MATCC
from src.master import MASTER
from my_lr_scheduler import ChainedScheduler

# os.environ['CUDA_VISIBLE_DEVICES']="7" # 程序可见的GPU
cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


class TrainConfig:
    model_name = "MATCC_csi800"
    GPU = 0
    universe = 'csi800'
    seed = 11033  # 11031.13031
    dataset_dir_path = "./dataset"
    model_save_path = f"./model_params/{universe}/{model_name}_{seed}"
    metrics_loss_path = f"./metrics/{universe}/{model_name}_{seed}"
    log_dir = f"./logs/{model_name}_{seed}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(metrics_loss_path):
        os.makedirs(metrics_loss_path)

    logging.basicConfig(filename=os.path.join(log_dir, f"{model_name}_seed_{seed}.log"), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(f"Train {model_name}")
    writer = SummaryWriter(
        log_dir=log_dir, filename_suffix=f"{model_name}_{seed}_")

    # 设置epoch, lr
    n_epoch = 75
    lr = 3e-4
    gamma = 1.0
    coef = 1.0
    cosine_period = 4
    T_0 = 15
    T_mult = 1
    warmUp_epoch = 10
    eta_min = 2e-5

    weight_decay = 0.001

    # 模型输入特征
    seq_len = 8
    d_feat = 158
    d_model = 256
    n_head = 4
    dropout = 0.5
    gate_input_start_index = 158
    gate_input_end_index = 221
    train_stop_loss_threshold = 0.95
    device = torch.device(
        f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    beta = 10 if universe == 'csi300' else 5
    # 模型初始化
    model = MATCC(d_model=d_model, d_feat=d_feat, seq_len=seq_len, t_nhead=n_head, S_dropout_rate=dropout).to(device)
    #model = MASTER(d_feat=d_feat, d_model=d_model, S_dropout_rate=dropout, T_dropout_rate=dropout, beta=beta).to(device)

    train_optimizer = optim.Adam(model.parameters(), lr=lr, betas=(
        0.9, 0.999), weight_decay=weight_decay)

    lr_scheduler = ChainedScheduler(train_optimizer, T_0=T_0, T_mul=T_mult, eta_min=eta_min,
                                    last_epoch=-1, max_lr=lr, warmup_steps=warmUp_epoch,
                                    gamma=gamma, coef=coef, step_size=3, cosine_period=cosine_period)

    writer.add_text("train_optimizer", train_optimizer.__str__())
    writer.add_text("lr_scheduler", lr_scheduler.__str__())

    logger.info(msg=f"\n===== Model {model_name} =====\n"
                    f"n_epochs: {n_epoch}\n"
                    f"start_lr: {lr}\n"
                    f"T_0: {T_0}\n"
                    f"T_mult: {T_mult}\n"
                    f"gamma: {gamma}\n"
                    f"coef: {coef}\n"
                    f"cosine_period: {cosine_period}\n"
                    f"eta_min: {eta_min}\n"
                    f"seed: {seed}\n"
                    f"optimizer: {train_optimizer}\n"
                    f"lr_scheduler: {lr_scheduler}\n"
                    f"description: train {model_name}\n\n")

    writer.add_text("model_name", model_name)
    writer.add_text("seed", str(seed))
    writer.add_text("n_head", str(n_head))
    writer.add_text("learning rate", str(lr))
    writer.add_text("T_0", str(T_0))
    writer.add_text("T_mult", str(T_mult))
    writer.add_text("gamma", str(gamma))
    writer.add_text("coef", str(coef))
    writer.add_text("eta_min", str(eta_min))
    writer.add_text("weight_decay", str(weight_decay))
    writer.add_text("cosine_period", str(cosine_period))


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


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    loss = (pred[mask] - label[mask]) ** 2
    return torch.mean(loss)


def _init_data_loader(data, shuffle=True, drop_last=True):
    sampler = DailyBatchSamplerRandom(data, shuffle)
    data_loader = DataLoader(
        data, sampler=sampler, drop_last=drop_last, num_workers=2, pin_memory=True)
    return data_loader


def train_epoch(data_loader, train_optimizer, lr_scheduler, model, device):
    model.train()
    losses = []

    for data in data_loader:
        data = torch.squeeze(data, dim=0)
        '''
        data.shape: (N, T, F)
        N - number of stocks
        T - length of lookback_window, 8
        F - 158 factors + 63 market information + 1 label           
        '''
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1].to(device)

        pred = model(feature.float())

        loss = loss_fn(pred, label)
        losses.append(loss.item())

        train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        train_optimizer.step()
    lr_scheduler.step()

    return float(np.mean(losses))


def valid_epoch(data_loader, model, device):
    model.eval()
    losses = []
    ic = []
    ric = []
    with torch.no_grad():
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(device)
            label = data[:, -1, -1].to(device)
            with torch.no_grad():
                pred = model(feature.float())
            loss = loss_fn(pred, label)
            losses.append(loss.item())

            daily_ic, daily_ric = calc_ic(
                pred.detach().cpu().numpy(), label.detach().cpu().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

    metrics = {
        'IC': np.mean(ic),
        'ICIR': np.mean(ic) / np.std(ic),
        'RIC': np.mean(ric),
        'RICIR': np.mean(ric) / np.std(ric)
    }

    return float(np.mean(losses)), metrics


def train():
    if not os.path.exists(TrainConfig.dataset_dir_path):
        raise FileExistsError("Data dir not exists")

    universe = TrainConfig.universe  # or 'csi800'

    # Please install qlib first before load the data.
    with open(f'{TrainConfig.dataset_dir_path}/{universe}/{universe}_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    with open(f'{TrainConfig.dataset_dir_path}/{universe}/{universe}_dl_valid.pkl', 'rb') as f:
        dl_valid = pickle.load(f)
    with open(f'{TrainConfig.dataset_dir_path}/{universe}/{universe}_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)
    print("Data Loaded.")

    # 核心代码

    train_loader = _init_data_loader(dl_train, shuffle=True, drop_last=True)
    valid_loader = _init_data_loader(dl_valid, shuffle=False, drop_last=False)
    test_loader = _init_data_loader(dl_test, shuffle=False, drop_last=False)

    device = TrainConfig.device
    writer = TrainConfig.writer

    # Model
    model = TrainConfig.model

    # LR

    # train_optimizer = optim.Adam(model.parameters(), lr=TrainConfig.lr, betas=(0.9, 0.999),
    #                              weight_decay=TrainConfig.weight_decay)
    train_optimizer = TrainConfig.train_optimizer
    lr_scheduler = TrainConfig.lr_scheduler

    best_valid_loss = np.Inf

    print("==" * 10 +
          f" Now is Training {TrainConfig.model_name}_{TrainConfig.seed} " + "==" * 10 + "\n")

    # 训练
    for step in range(TrainConfig.n_epoch):
        train_loss = train_epoch(train_loader, train_optimizer=train_optimizer, lr_scheduler=lr_scheduler, model=model,
                                 device=device)
        val_loss, valid_metrics = valid_epoch(valid_loader, model, device)
        test_loss, test_metrics = valid_epoch(test_loader, model, device)

        if writer is not None:
            writer.add_scalars(
                "Valid metrics", valid_metrics, global_step=step)
            writer.add_scalars("Test metrics", test_metrics, global_step=step)
            writer.add_scalar("Train loss", train_loss, global_step=step)
            writer.add_scalar("Valid loss", val_loss, global_step=step)
            writer.add_scalar("Test loss", test_loss, global_step=step)
            writer.add_scalars("All loss Comparison",
                               {"train loss": train_loss,
                                   "val loss": val_loss, "test loss": test_loss},
                               global_step=step)
            writer.add_scalar(
                "Learning rate", train_optimizer.param_groups[0]['lr'], global_step=step)

        print(
            "==" * 10 + f" {TrainConfig.model_name}_{TrainConfig.seed} Epoch {step} " + "==" * 10)
        print("Epoch %d, train_loss %.6f, valid_loss %.6f, test_loss %.6f " %
              (step, train_loss, val_loss, test_loss))
        print("Valid Dataset Metrics performance:{}\n".format(valid_metrics))
        print("Test Dataset Metrics performance:{}\n".format(test_metrics))
        print("Learning rate :{}\n\n".format(
            train_optimizer.param_groups[0]['lr']))

        TrainConfig.logger.info(msg=f"\n===== Epoch {step} =====\ntrain loss:{train_loss}, "
                                    f"valid loss:{val_loss},test loss:{test_loss}\n"
                                    f"valid metrics:{valid_metrics}\n"
                                    f"test metrics:{test_metrics}\n"
                                    f"learning rate:{train_optimizer.param_groups[0]['lr']}\n")

        # 保存参数,新的保存策略,只保留18以上的结果
        if step <= 10:
            continue

        if (step-10) % 15 == 0:
            best_valid_loss = val_loss
            model_param = copy.deepcopy(model.state_dict())
            torch.save(model_param,
                       f'{TrainConfig.model_save_path}/{TrainConfig.model_name}_model_params_epoch_{step}_seed_{TrainConfig.seed}.pth')

    print("SAVING LAST EPOCH RESULT AS THE TEST RESULT!")
    torch.save(model.state_dict(),
               f'{TrainConfig.model_save_path}/TEST_{TrainConfig.model_name}_model_params_seed_{TrainConfig.seed}.pth')
    print("\n" + "==" * 10 + " Training Over " + "==" * 10)
    writer.close()


if __name__ == "__main__":
    train()
