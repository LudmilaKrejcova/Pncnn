"""Pilot pNCNN training for prepared tire NPZ tiles; run from the repository root."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from common.unet import UNetSP
from tireNconv import NConvUNet


class TireModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Explicit validity is an additional input to the confidence estimator.
        self.confEstimator = UNetSP(2, 1)
        self.nconv = NConvUNet(1, 1)
        self.precisionEstimator = UNetSP(1, 1)

    def forward(self, depth, valid):
        confidence = self.confEstimator(torch.cat((depth, valid.float()), 1))
        confidence = confidence * valid.float()
        prediction, propagated = self.nconv(depth, confidence)
        precision = self.precisionEstimator(propagated).clamp(1e-6, 1e6)
        return prediction, precision


class TireDataset(Dataset):
    def __init__(self, root, records, normalization):
        self.root = root
        self.records = records
        self.mean = float(normalization['mean'])
        self.scale = float(normalization['recommendedScale'])
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid normalization scale.')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with np.load(self.root / self.records[index]['tilePath'], allow_pickle=False) as tile:
            padding = tile['paddingMask'].astype(bool)
            valid = tile['inputValidMask'].astype(bool) & ~padding
            targetValid = tile['targetValidMask'].astype(bool) & ~padding
            if np.any(targetValid & ~np.isin(tile['targetSource'], [1, 2, 3])):
                raise ValueError('Untrusted target included in targetValidMask.')
            completion = targetValid & ~valid
            if not np.array_equal(completion, tile['completionMask']):
                raise ValueError('Inconsistent completion mask.')
            depth = tile['inputDepth']
            target = tile['targetDepth']
            if not np.isfinite(depth[valid]).all() or not np.isfinite(target[targetValid]).all():
                raise ValueError('Nonfinite valid depth.')
            if not targetValid.any():
                raise ValueError('Tile has no valid targets.')
            depth = np.where(valid, (depth - self.mean) / self.scale, 0).astype('float32')
            target = np.where(targetValid, (target - self.mean) / self.scale, 0).astype('float32')
            return tuple(torch.from_numpy(a.copy()).unsqueeze(0) for a in
                         (depth, valid, target, targetValid, completion))


def maskedLoss(prediction, precision, target, targetValid):
    residual = prediction[targetValid] - target[targetValid]
    weight = precision[targetValid]
    return (weight * residual.square() - weight.log()).mean()


def savePreview(batch, prediction, destination, normalization):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    depth, valid, target, targetValid, completion = [a[0, 0].detach().cpu().numpy() for a in batch]
    pred = prediction[0, 0].detach().cpu().numpy()
    scale, mean = normalization['recommendedScale'], normalization['mean']
    depth, target, pred = [a * scale + mean for a in (depth, target, pred)]
    lo, hi = np.percentile(target[targetValid], [1, 99])
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for ax, array, title in zip(axes[:3], [np.where(valid, depth, np.nan), pred,
                               np.where(targetValid, target, np.nan)],
                               ['Input (mm)', 'Prediction (mm)', 'Fusion pseudo-GT (mm)']):
        ax.imshow(array, vmin=lo, vmax=hi, cmap='viridis'); ax.set_title(title)
    axes[3].imshow(completion, cmap='gray'); axes[3].set_title('Evaluated holes')
    im = axes[4].imshow(np.where(completion, np.abs(pred-target), np.nan), cmap='magma')
    axes[4].set_title('Hole error (mm)'); fig.colorbar(im, ax=axes[4])
    for ax in axes: ax.axis('off')
    fig.tight_layout(); fig.savefig(destination, dpi=130); plt.close(fig)


def runEpoch(model, loader, device, scale, optimizer=None, previewPath=None, normalization=None):
    training = optimizer is not None
    model.train(training)
    sums = {'all': [0., 0., 0], 'holes': [0., 0., 0]}
    lossSum, pixels = 0., 0
    started = time.time()
    for step, batch in enumerate(loader, 1):
        batch = tuple(a.to(device) for a in batch)
        depth, valid, target, targetValid, completion = batch
        with torch.set_grad_enabled(training):
            prediction, precision = model(depth, valid)
            loss = maskedLoss(prediction, precision, target, targetValid)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; training stopped.')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
        count = int(targetValid.sum())
        lossSum += float(loss.detach()) * count; pixels += count
        error = (prediction.detach() - target) * scale
        for key, mask in [('all', targetValid), ('holes', completion)]:
            values = error[mask].double()
            sums[key][0] += float(values.abs().sum())
            sums[key][1] += float(values.square().sum())
            sums[key][2] += values.numel()
        if step == 1 and previewPath is not None:
            savePreview(batch, prediction, previewPath, normalization)
        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f"{'Train' if training else 'Val'} {step}/{len(loader)} loss={loss.item():.5f} elapsed={time.time()-started:.1f}s", flush=True)
    result = {'loss': lossSum / pixels}
    for key, (absolute, squared, count) in sums.items():
        result[key+'Count'] = count
        result[key+'MaeMm'] = absolute/count if count else None
        result[key+'RmseMm'] = (squared/count)**0.5 if count else None
    return result


def selectRecords(records, split, limit, seed):
    selected = [r for r in records if r['split'] == split]
    random.Random(seed).shuffle(selected)
    return selected[:limit] if limit else selected


def checkSplits(records):
    tireSplits = {}
    for record in records:
        tireSplits.setdefault(record['tireId'], set()).add(record['split'])
    if any(len(splits) != 1 for splits in tireSplits.values()):
        raise ValueError('A tire occurs in multiple splits.')
    # Evaluation tiles must not overlap within the same scan and camera.
    groups = {}
    for r in records:
        if r['split'] not in ('val', 'test'): continue
        key = (r['split'], r['scanId'], r['inputCamera'], r.get('region'))
        rectangle = (r['row'], r['column'], r['row']+r['realHeight'], r['column']+r['realWidth'])
        for other in groups.setdefault(key, []):
            if rectangle[0] < other[2] and other[0] < rectangle[2] and rectangle[1] < other[3] and other[1] < rectangle[3]:
                raise ValueError('Overlapping evaluation tiles require reconstruction before scoring.')
        groups[key].append(rectangle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/tirePilot'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', dest='batchSize', type=int, default=1)
    parser.add_argument('--train-limit', dest='trainLimit', type=int, default=0)
    parser.add_argument('--val-limit', dest='valLimit', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batchSize < 1 or min(args.trainLimit, args.valLimit) < 0 or args.lr <= 0:
        parser.error("Epochs and batch size must be positive; limits must be nonnegative; learning rate must be positive.")
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use the project Python environment or --device cpu.')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError('Output directory is not empty. Choose a new output or use --resume.')
    normalization = json.loads((args.data/'normalization.json').read_text(encoding='utf-8-sig'))
    if normalization.get('appliedToNPZ'):
        raise ValueError('This loader expects raw depths in mm.')
    records = [json.loads(line) for line in (args.data/'tiles.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    checkSplits(records)
    trainRecords = selectRecords(records, 'train', args.trainLimit, args.seed)
    valRecords = selectRecords(records, 'val', args.valLimit, args.seed)
    if not trainRecords or not valRecords: raise ValueError('Train and validation must be nonempty.')
    model = TireModel().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    startEpoch, best = 0, float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        if checkpoint['normalization'] != normalization: raise ValueError('Normalization differs from checkpoint.')
        if checkpoint['trainPaths'] != [r['tilePath'] for r in trainRecords] or checkpoint['valPaths'] != [r['tilePath'] for r in valRecords]:
            raise ValueError('Resume requires the same data subset and seed.')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        startEpoch, best = checkpoint['epoch'], checkpoint['best']
        torch.set_rng_state(checkpoint['rng'].cpu())
        if args.device == 'cuda' and checkpoint.get('cudaRng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cudaRng']])
    args.output.mkdir(parents=True, exist_ok=True)
    loaders = [DataLoader(TireDataset(args.data, rows, normalization), batch_size=args.batchSize,
                          shuffle=shuffle, num_workers=0) for rows, shuffle in [(trainRecords, True), (valRecords, False)]]
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(normalization=normalization, trainTiles=len(trainRecords), valTiles=len(valRecords),
                  architecture='pNCNN with explicit mask input and masked initial confidence',
                  evaluation='Pixel-weighted, per camera; nonoverlapping validation tiles only; pseudo-GT.')
    (args.output/'settings.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Device: {args.device}; train: {len(trainRecords)}; val: {len(valRecords)}', flush=True)
    for epoch in range(startEpoch+1, args.epochs+1):
        print(f'Epoch {epoch}/{args.epochs}', flush=True)
        train = runEpoch(model, loaders[0], args.device, normalization['recommendedScale'], optimizer)
        val = runEpoch(model, loaders[1], args.device, normalization['recommendedScale'],
                       previewPath=args.output/f'epoch{epoch:03d}.png', normalization=normalization)
        score = val['holesMaeMm'] if val['holesMaeMm'] is not None else val['allMaeMm']
        improved = score < best; best = min(best, score)
        checkpoint = dict(epoch=epoch, best=best, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          normalization=normalization, config=config, rng=torch.get_rng_state(),
                          cudaRng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None,
                          trainPaths=[r['tilePath'] for r in trainRecords], valPaths=[r['tilePath'] for r in valRecords])
        torch.save(checkpoint, args.output/'last.tmp'); (args.output/'last.tmp').replace(args.output/'last.pt')
        if improved: torch.save(checkpoint, args.output/'best.pt')
        row = {'epoch': epoch, **{'train_'+k:v for k,v in train.items()}, **{'val_'+k:v for k,v in val.items()}}
        path = args.output/'metrics.csv'; exists = path.exists()
        with path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists: writer.writeheader()
            writer.writerow(row)
        print('Validation:', json.dumps(val), flush=True)
    print('Finished. Check metrics.csv, best.pt and epoch previews.', flush=True)


if __name__ == '__main__':
    main()
