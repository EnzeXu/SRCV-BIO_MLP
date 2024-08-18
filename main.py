import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import sys
import wandb
from timeit import default_timer

from fno1d import MyFNO1d, MLP
from utils import *
from utilities3 import count_params, LpLoss


def main(opt):
    gpu_id = opt.gpu_id
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda', gpu_id)
    else:
        device = torch.device('cpu')
    print("using {}".format(device))
    set_random_seeds(opt.seed)
    print(f"seed: {opt.seed}")

    batch_size = 64
    learning_rate = 0.001

    epochs = opt.epoch  # default 500
    step_size = 100  # default 100
    gamma = 0.5

    modes = 16
    width = 64

    ntrain = 8000
    ntest = 2000
    s = 100
    sub = 100000 // s
    n_dim = 2

    log_path = f"output/summary/{opt.model}.csv"
    if not os.path.exists(log_path):
        print(f"file {log_path} does not exist. Creating title row ...")
        with open(log_path, "w") as f:
            f.write(
                f"start_time,end_time,model,epoch,gpu_id,norm,time_cost,seed,best_test_l2_epoch,best_test_l2\n")

    data: np.ndarray = np.load(opt.data_path)
    print(f"raw data shape: {data.shape}")
    data: torch.Tensor = torch.tensor(data, dtype=torch.float32).to(device)

    x_data = data[:, :, :n_dim]
    y_data = data[:, :, -n_dim:]

    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    grid_all = np.linspace(0, 1, 100000).reshape(100000, 1).astype(np.float64)
    grid = grid_all[::sub, :]
    grid = torch.tensor(grid, dtype=torch.float).to(device)
    x_train = torch.cat([x_train.reshape(ntrain, s, n_dim), grid.repeat(ntrain, 1, 1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest, s, n_dim), grid.repeat(ntest, 1, 1)], dim=2)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    if opt.model == "FNO":
        model = MyFNO1d(modes, width).to(device)
    elif opt.model == "MLP":
        model = MLP(n_dim, n_dim).to(device)
    else:
        model = MLP(n_dim + 1, n_dim).to(device)

    print(f"Model Parameter: {count_params(model)}")
    print(f"Model:")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    start_time = time.time()
    myloss = LpLoss(size_average=False)
    # y_normalizer.cuda()
    best_test_l2 = float("inf")
    best_test_l2_epoch = -1
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            # print(f"x shape: {x.shape}, out shape: {out.shape}")

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse.backward()
            # out = y_normalizer.decode(out.view(batch_size, -1))
            # y = y_normalizer.decode(y)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            # l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                # out = y_normalizer.decode(out.view(batch_size, -1))
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()

        print("Epoch: %d, time: %.3f, Train Loss: %.12f, Train l2: %.12f, Test l2: %.12f, lr: %.6f"
              % (ep, t2 - t1, train_mse, train_l2, test_l2, optimizer.param_groups[0]["lr"]) + (" [updated save]" if test_l2 < best_test_l2 else ""))
        if test_l2 < best_test_l2:
            torch.save(model.state_dict(), f'saves/{opt.timestring}_{opt.model.lower()}_best.pt')
            best_test_l2 = test_l2
            best_test_l2_epoch = ep + 1
        if opt.wandb:
            try:
                wandb.log({
                    'epoch': ep + 1,
                    'train_loss': train_l2,
                    'val_loss': test_l2,
                    'lr': optimizer.param_groups[0]["lr"],
                })
            except Exception as e:
                print("[Error]", e)
                pass
        # print(ep, t2-t1, train_mse, train_l2, test_l2)
    torch.save(model.state_dict(), f'saves/{opt.timestring}_{opt.model.lower()}_last.pt')
    print(f"best test_l2: {best_test_l2}")

    elapsed = time.time() - start_time
    print("\n=============================")
    print("Training done...")
    print('Training time: %.3f' % (elapsed))
    print("=============================\n")
    end_time = get_timestring()
    with open(log_path, "a") as f:
        f.write(f"{opt.timestring},{end_time},{opt.model},{opt.epoch},{opt.gpu_id},{opt.data_norm},{elapsed},{opt.seed},{best_test_l2_epoch},{best_test_l2}\n")

    return model


if __name__ == '__main__':
    print(f"Arguments: {' '.join(sys.argv)}")
    parser = argparse.ArgumentParser(description='MLP/FNO training')
    parser.add_argument("--gpu_id", default=0, type=int, choices=[0, 1, 2, 3], help="""gpu_id""")
    parser.add_argument("--wandb", default=True, type=bool, help="""use wandb""")
    parser.add_argument("--model", default="MLP", choices=["MLP", "MLP_with_grid", "FNO"], type=str, help="""model type""")
    parser.add_argument("--seed", default=42, type=int, help="""random seed""")
    parser.add_argument("--epoch", default=1000, type=int, help="""epoch""")
    parser.add_argument("--data_norm", default=False, type=bool, help="""data_norm""")
    parser.add_argument('--timestring', type=str, default="", help="""timestring""")
    opt = parser.parse_args()

    if opt.timestring == "":
        opt.timestring = get_timestring()
    else:
        assert len(opt.timestring) == 22

    print(f"Timestring: {opt.timestring}")
    if opt.data_norm:
        opt.data_path = "data/MPF_2_separate_0.000001_norm.npy"
    else:
        opt.data_path = "data/MPF_2_separate_0.000001.npy"

    if opt.wandb:
        with wandb.init(project=f"SRCV-BIO_{opt.model}", name=f"{opt.timestring}"):
            main(opt)
    else:
        main(opt)
    pass
