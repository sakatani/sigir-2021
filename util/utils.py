import gc
import random

import numpy as np
import torch
import tqdm

def reduce_df(df):
    for i, c in enumerate(df.columns):
        dtype = df.dtypes[i]  # df[c].dtypes
        if dtype == 'uint8':
            df[c] = df[c].astype(np.int8)
            gc.collect()
        elif dtype == 'bool':
            df[c] = df[c].astype(np.int8)
            gc.collect()
        elif dtype == 'uint32':
            df[c] = df[c].astype(np.int32)
            gc.collect()
        elif dtype == 'int64':
            df[c] = df[c].astype(np.int32)
            gc.collect()
        elif dtype == 'float64':
            df[c] = df[c].astype(np.float32)
            gc.collect()
    return df

def seed_torch(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(model, epoch, optimizer, scheduler, scaler, best_score, fold, seed, fname):
    checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_score': best_score,
        }
    torch.save(checkpoint, './checkpoints/%s_%d_%d.pt' % (fname, fold, seed))


def train_epoch(loader, model, optimizer, scheduler, device):
    model.train()
    model.zero_grad()
    losses = []
    bar = tqdm(iter(loader))

    for batch in bar:
        batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys()}
        out_dict = model(batch)
        loss_torch = out_dict['loss']
        loss = loss_torch.detach().cpu().numpy()
        losses.append(loss)
        smooth_loss = sum(losses[-100:]) / min(len(losses), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss, smooth_loss))

        loss_torch.backward()
        del loss_torch, out_dict
        optimizer.step()
        scheduler.step()
        for p in model.parameters():
            p.grad = None

    del batch, loss
    _ = gc.collect()
    torch.cuda.empty_cache()
    return losses


def val_epoch(loader, model, device):
    model.eval()
    losses = []
    preds = []
    labels = []
    bar = tqdm(iter(loader))

    with torch.no_grad():
        for batch in bar:
            batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys()}

            out_dict = model(batch)
            loss = out_dict['loss'].detach().cpu().numpy()
            preds.append(out_dict['logits'].detach())
            labels.append(batch['target'].detach())
            losses.append(loss)

            smooth_loss = sum(losses[-100:]) / min(len(losses), 100)
            bar.set_description('loss: %.5f, smth: %.5f' % (loss, smooth_loss))

            del out_dict, loss
            torch.cuda.empty_cache()

        loss_mean = np.mean(losses)

    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    del batch, bar
    _ = gc.collect()

    return loss_mean, preds, labels


def test_epoch(loader, models, device):
    preds = []
    with torch.no_grad():
        bar = tqdm(range(len(loader)))
        for batch in bar:
            batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys()}
            pred = 0
            for model in models:
                out_dict = model(batch)
                pred = pred + out_dict['logits'] / len(models)
            preds.append(pred.detach())
    return preds


def get_topk(preds, k=20):
    return preds.argsort(axis=1)[:,-k:][:,::-1]

def mrr_at_k(preds, labels, k=20):
    assert len(labels) > 0
    assert len(preds) == len(labels)

    # get top K predictions
    topk = get_topk(preds, k=k)
    rr = []
    for p, l in zip(topk, labels):
        if len(l) == 0:
            rr.append(0.0)
        else:
            # get next_item from labels
            next_item = l[0]
            # add 0.0 explicitly if not there (for transparency)
            if next_item not in p:
                rr.append(0.0)
            # else, take the reciprocal of prediction rank
            else:
                rr.append(1.0 / (p.index(next_item) + 1))

    # return the mean reciprocal rank
    return sum(rr) / len(labels)