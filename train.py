import torch
from tqdm import tqdm
import os
import numpy as np

from util import save_midi

def train_one_step(args, model, optimizer, real_samples, loss_fn):
    input = torch.cat([real_samples[:,:,:16,:], real_samples[:,:,-16:,:]], dim = 2)


    real_samples = real_samples.to(args.device)
    input = input.to(args.device)

    optimizer.zero_grad()

    pred = model(input)
    loss = loss_fn(pred, real_samples)

    loss.backward()
    optimizer.step()

    return loss, pred, real_samples

def train(args, model, dataloader, cur_epoch = 1):
    model.train()
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.5, 0.99))
    pbar = tqdm(range(cur_epoch, args.n_epochs+1), desc="Training", unit = "epoch")
    for epoch in pbar:
        for idx, real_samples in enumerate(dataloader):
            loss, pred, real = train_one_step(args, model, optimizer, real_samples[0], loss_fn)
            pbar.set_postfix_str("loss: {:.3f}".format(loss))
        epoch += 1
        pbar.update(1)

        if epoch % args.ckpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.ckpoint, "epoch{}.pt".format(epoch)))
            if args.save_sample:
                prediction = pred[0].detach().cpu()
                save_midi(os.path.join(args.ckpoint, "sample", "pred_epoch{}.mid".format(epoch)), prediction)
                target = real[0].detach().cpu()
                save_midi(os.path.join(args.ckpoint, "sample", "target_epoch{}.mid".format(epoch)), target)
    pbar.close()

def mix(args, model, mid1, mid2, start1, start2):
    """
    mid1, mid2: (n_tracks, length, n_pitches) array
    start1, start2: mixed point of each midi
    """
    model.eval()
    input = np.concatenate((mid1[:,:,start1:start1+16,:], mid2[:,:,start2:start2+16,:]), axis = -2)
    input = torch.as_tensor(input, dtype=torch.float32, device = args.device)
    pred = model(input).detach().cpu()
    return
