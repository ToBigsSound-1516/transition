import torch
from tqdm import tqdm
import os
import numpy as np
import time

from util import save_midi, midi_to_array

def train_one_step(args, model, optimizer, real_samples, loss_fn):
    inputs = torch.cat([real_samples[:,:,:16,:], real_samples[:,:,-16:,:]], dim = 2).clone().detach()

    if not args.cpu and torch.cuda.is_available():
        real_samples = real_samples.cuda()
        inputs = inputs.cuda()

    optimizer.zero_grad()

    pred = model(inputs)
    loss = loss_fn(pred, real_samples) / real_samples.shape[0]

    loss.backward()
    optimizer.step()
    # scheduler.step()

    return loss.item(), pred.detach().cpu(), real_samples.detach().cpu()

def train(args, model, dataloader, cur_epoch = 1):
    loss_fn = torch.nn.MSELoss(reduction = "sum")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.5, 0.99))
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    pbar = tqdm(initial= cur_epoch, total = args.n_epochs, desc="Training", unit = "epoch")
    for epoch in range(cur_epoch, args.n_epochs+1):
        for idx, real_samples in enumerate(dataloader):
            model.train()
            loss, pred, real = train_one_step(args, model, optimizer, real_samples[0], loss_fn)
            pbar.set_postfix_str("loss: {:.3f}".format(loss))

        log_name = os.path.join(args.ckpoint, 'loss_log.txt')
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write("[{}]\tepoch: {:04d}\tloss: {:.3f}\n".format(now, epoch, loss))

        if epoch % args.ckpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.ckpoint, "epoch{:04d}.pt".format(epoch)))
            if args.save_sample:
                prediction = pred[0].detach().cpu()
                save_midi(os.path.join(args.ckpoint, "sample", "pred_epoch{:04d}.mid".format(epoch)), prediction)
                target = real[0].detach().cpu()
                save_midi(os.path.join(args.ckpoint, "sample", "target_epoch{:04d}.mid".format(epoch)), target)
        pbar.update(1)
    pbar.close()

def mix_arr(args, model, mid1, mid2, start1, start2, margin):
    """
    mid1, mid2: (n_tracks, length, n_pitches) array
    start1, start2: mixed point of each midi
    margin: if margin is -1, it will cover whole song.
    """
    model.eval()
    input = np.concatenate((mid1[:,start1:start1+16,:], mid2[:,start2:start2+16,:]), axis = -2)
    input = torch.as_tensor(input, dtype=torch.float32, device = args.device).unsqueeze(0)
    pred = model(input).detach().cpu().squeeze()
    if margin == -1:
        mixed = np.concatenate((mid1[:, :start1, :], pred, mid2[:, start2 + 16:, :]),
                               axis=-2)
    else:
        mixed = np.concatenate((mid1[:,start1-margin:start1,:], pred, mid2[:,start2+16:start2+16+margin,:]), axis = -2)
    return mixed

def mix(args, model):
    mid1 = midi_to_array(args.midi_path1)
    mid2 = midi_to_array(args.midi_path2)
    mixed = mix_arr(args, model, mid1, mid2, args.start1, args.start2, args.mix_margin)
    if args.midi_save_dir.endswith(".mid"):
        save_midi(args.midi_save_dir, mixed)
        mixed_name = os.path.basename(args.midi_save_dir)
    else:
        mixed_name = "{}+{}.mid".format(os.path.basename(args.midi_path1).split(".")[0], os.path.basename(args.midi_path2).split(".")[0])
        save_midi(os.path.join(args.midi_save_dir, mixed_name), mixed)
    print("{} is saved in {}".format(mixed_name, args.midi_save_dir))
