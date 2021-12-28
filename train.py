import torch
from tqdm import tqdm
import os

from util import save_midi

def train_one_step(args, model, optimizer, real_samples, loss_fn):
    """Train the networks for one step."""
    input = torch.cat([real_samples[:,:,:16,:], real_samples[:,:,-16:,:]], dim = 2)


    real_samples = real_samples.to(args.device)
    input = input.to(args.device)

    optimizer.zero_grad()

    pred = model(input)
    loss = loss_fn(pred, real_samples)

    loss.backward()
    optimizer.step()

    return loss, pred, real_samples

def train(args, model, dataloader, step = 1):
    model.train()
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.5, 0.99))
    pbar = tqdm(desc="Training", initial=step, total=args.n_steps)
    while step < args.n_steps+1:
        for idx, real_samples in enumerate(dataloader):
            if step > args.n_steps + 1:
                break

            loss, pred, real = train_one_step(args, model, optimizer, real_samples[0], loss_fn)
            pbar.set_postfix_str("loss: {:.3f}".format(loss))
            pbar.update(1)
            step += 1
            if step % args.sample_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.ckpoint, "step{}.pt".format(step)))
                if args.save_sample:
                    prediction = pred[0].detach().cpu()
                    save_midi(os.path.join(args.ckpoint, "sample", "pred_step{}.mid".format(step)), prediction)
                    target = real[0].detach().cpu()
                    save_midi(os.path.join(args.ckpoint, "sample", "target_step{}.mid".format(step)), target)
    pbar.close()

