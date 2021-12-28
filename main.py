import argparse
import os
import torch
import numpy as np

from model import Model
from util import get_latest_ckpoint
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default = 128)
    parser.add_argument('--ckpoint', type=str, default = "./ckpoint")
    parser.add_argument('--data_path', type=str, default = "./data")
    parser.add_argument('--train', action='store_true', help='enables training')
    parser.add_argument('--lr', type=float, default = 1e-5)
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--n_steps', type=int, default=100000)
    parser.add_argument('--sample_interval', type=int, default=10000)
    parser.add_argument('--save_sample', action='store_true', help="save some midi samples")
    parser.add_argument('--cpu', action='store_true', help='enables CUDA training')
    args = parser.parse_args()

    model = Model(args.latent_dim)
    print("="*80)
    print("Model is Created. Number of parameters: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if not args.cpu and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    os.makedirs(args.ckpoint, exist_ok=True)
    if args.save_sample:
        os.makedirs(os.path.join(args.ckpoint, "sample"), exist_ok= True)

    ck_path = get_latest_ckpoint(args.ckpoint)
    step = 1
    if ck_path is not None:
        step = int(ck_path.split("step")[-1].split(".")[0])
        model.load_state_dict(torch.load(ck_path))
        print("{} is loaded.".format(ck_path))
    model = model.to(args.device)

    if args.train:
        print("=" * 80)
        print("Prepare to train...")
        assert os.path.exists(os.path.join(args.data_path, "data.npy")), "There is no data.npy in the "+args.data_path
        data = np.load(os.path.join(args.data_path, "data.npy"))
        print("Data is loaded. Data shape:", data.shape)
        data = torch.as_tensor(data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, drop_last=True, shuffle=False)
        train(args, model, data_loader, step)




