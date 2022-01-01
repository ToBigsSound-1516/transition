import argparse
import os
import torch
import numpy as np

from model import Model
from util import get_latest_ckpoint
from train import train, mix
from const import measure_resolution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default = 128)
    parser.add_argument('--ckpoint', type=str, default = "./ckpoint")
    parser.add_argument('--cpu', action='store_true', help='enables CUDA training')

    # Training parameter
    parser.add_argument('--train', action='store_true', help='enables training')
    parser.add_argument('--data_path', type=str, default = "./data")
    parser.add_argument('--lr', type=float, default = 1e-5)
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--ckpoint_interval', type=int, default=10)
    parser.add_argument('--save_sample', action='store_true', help="save some midi samples")

    # Mixing parameter
    parser.add_argument('--midi_path1', type=str)
    parser.add_argument('--midi_path2', type=str)
    parser.add_argument('--midi_save_dir', type=str, default="./data/mixed")
    parser.add_argument('--start1', type=int)
    parser.add_argument('--start2', type=int)
    parser.add_argument('--mix_margin', type=int, default=4 * measure_resolution)
    args = parser.parse_args()

    model = Model(args.latent_dim)
    print("="*80)
    print("Model is Created. Number of parameters: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if not args.cpu and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    if args.ckpoint.endswith(".pt"):
        ck_path = args.ckpoint
        args.ckpoint = os.path.dirname(args.ckpoint)
    else:
        os.makedirs(args.ckpoint, exist_ok=True)
        ck_path = get_latest_ckpoint(args.ckpoint)

    if args.save_sample:
        os.makedirs(os.path.join(args.ckpoint, "sample"), exist_ok=True)

    step = 1
    if ck_path is not None:
        step = int(ck_path.split("epoch")[-1].split(".")[0])
        model.load_state_dict(torch.load(ck_path))
        print("{} is loaded.".format(ck_path))
    else:
        assert args.train, "There is no model in "+args.ckpoint

    model = model.to(args.device)

    print("=" * 80)
    if args.train:
        print("Prepare to train...")
        assert os.path.exists(os.path.join(args.data_path, "data.npy")), "There is no data.npy in the "+args.data_path
        data = np.load(os.path.join(args.data_path, "data.npy"))

        print("Data is loaded. Data shape:", data.shape)
        data = torch.as_tensor(data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, drop_last=True, shuffle=False)

        train(args, model, data_loader, step)

    else:
        print("Prepare to mix...")
        assert os.path.exists(args.midi_path1), "midi file 1 does not exist."
        assert os.path.exists(args.midi_path2), "midi file 2 does not exist."
        assert args.start1 is not None and args.start2 is not None, "Mixed point is not given."
        if args.midi_save_dir.endswith(".mid"):
            os.makedirs(os.path.dirname(args.midi_save_dir), exist_ok=True)
        else:
            os.makedirs(args.midi_save_dir, exist_ok=True)
        mix(args, model)
