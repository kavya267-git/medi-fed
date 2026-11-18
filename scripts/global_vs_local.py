"""Each hospital's own model vs the global (FedAvg) model, scored on that hospital's own held-out patients.

Usage (repo root): OMP_NUM_THREADS=1 python scripts/global_vs_local.py MODALITY ROUNDS SEEDS OUT.json [CAP]
CAP keeps only CAP training records per hospital (the small-hospital test).
Diabetes reads the real PIMA split from /tmp/pimatest (scripts/prepare_real_data.py diabetes --pima).
"""
import json, sys, time, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'backend')
from ml.registry import get_adapter
from ml.common.types import TrainConfig
from ml.training.local_trainer import train_on_local_data, load_training_splits
from ml.simulation.hospitals import SIMULATED_HOSPITALS, generate_hospital_data
from app.federated.fedavg import ClientUpdate, federated_average

modality, rounds, seeds, out = sys.argv[1], int(sys.argv[2]), [int(s) for s in sys.argv[3].split(',')], sys.argv[4]
ad = get_adapter(modality, 'pytorch')
weighting = 'balanced' if modality in ('ehr', 'diabetes') else 'none'

CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # optional: keep only CAP training records per hospital

def shrink(src, seed):
    import pandas as pd
    dst = Path(f'/tmp/gvl_{modality}_cap{CAP}_{seed}')
    if dst.exists():
        return dst
    shutil.copytree(src, dst)
    rng = np.random.default_rng(seed)
    for h in dst.iterdir():
        d = h / modality
        if (d / 'train.csv').exists():
            df = pd.read_csv(d / 'train.csv'); df.iloc[np.sort(rng.permutation(len(df))[:CAP])].to_csv(d / 'train.csv', index=False)
        else:
            z = dict(np.load(d / 'train.npz', allow_pickle=True)); n = len(next(iter(z.values())))
            idx = np.sort(rng.permutation(n)[:CAP])
            np.savez(d / 'train.npz', **{k: (v[idx] if len(v) == n else v) for k, v in z.items()})
        m = json.load(open(d / 'manifest.json')); m['splits']['train'] = min(CAP, m['splits']['train'])
        json.dump(m, open(d / 'manifest.json', 'w'))
    return dst

def data_root(seed):
    r = _data_root(seed)
    return shrink(r, seed) if CAP else r

def _data_root(seed):
    if modality == 'diabetes':
        return Path('/tmp/pimatest')  # real PIMA split (prepare_real_data.py diabetes --pima)
    root = Path(f'/tmp/gvl_{modality}_{seed}')
    if not root.exists():
        for h in SIMULATED_HOSPITALS:
            generate_hospital_data(root, h, seed=seed, modalities=[modality])
    return root

def macro_f1(y, pred, k):
    f = []
    for c in range(k):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def pooled(root):
    import pandas as pd
    dst = Path(str(root) + '_pooled') / 'all' / modality
    if dst.exists():
        return dst.parent
    dst.mkdir(parents=True)
    hs = sorted(p for p in root.iterdir() if p.is_dir())
    for split in ('train', 'val'):
        if (hs[0] / modality / f'{split}.csv').exists():
            pd.concat([pd.read_csv(h / modality / f'{split}.csv') for h in hs]).to_csv(dst / f'{split}.csv', index=False)
        else:
            zs = [dict(np.load(h / modality / f'{split}.npz', allow_pickle=True)) for h in hs]
            np.savez(dst / f'{split}.npz', **{k: np.concatenate([z[k] for z in zs]) for k in zs[0]})
    m = json.load(open(hs[0] / modality / 'manifest.json'))
    m['splits'] = {'train': sum(json.load(open(h / modality / 'manifest.json'))['splits']['train'] for h in hs),
                   'val': sum(json.load(open(h / modality / 'manifest.json'))['splits']['val'] for h in hs)}
    json.dump(m, open(dst / 'manifest.json', 'w'))
    return dst.parent

results = []
for seed in seeds:
    root = data_root(seed); hs = sorted(p.name for p in root.iterdir() if p.is_dir())
    vals = {h: load_training_splits(root / h, modality)[1] for h in hs}
    trains = {h: len(load_training_splits(root / h, modality)[0]) for h in hs}
    w0 = ad.init_weights(seed); g = w0; local = {h: w0 for h in hs}; central = w0; proot = pooled(root)
    t = time.time()
    for r in range(rounds):
        cfg = TrainConfig(2, 32, 0.003, seed * 1000 + r, class_weighting=weighting)
        ups = []
        for h in hs:
            o = train_on_local_data(root / h, modality, g, cfg, 'pytorch'); ups.append(ClientUpdate(h, o.weights, o.num_samples))
            local[h] = train_on_local_data(root / h, modality, local[h], cfg, 'pytorch').weights
        g = federated_average(ups)
        central = train_on_local_data(proot, modality, central, cfg, 'pytorch').weights
        print(modality, 'seed', seed, 'round', r + 1, f'{time.time() - t:.0f}s', flush=True)
    k = ad.task.num_classes
    ys, pg, pl = [], [], []
    for h in hs:
        v = vals[h]
        p_g = ad.predict_proba(g, v.X).argmax(1); p_l = ad.predict_proba(local[h], v.X).argmax(1)
        results.append(dict(seed=seed, hospital=h, train=trains[h], held_out=len(v), own=macro_f1(v.y, p_l, k),
                            glob=macro_f1(v.y, p_g, k), own_acc=float((p_l == v.y).mean()), glob_acc=float((p_g == v.y).mean())))
        ys.append(v.y); pg.append(p_g); pl.append(p_l)
        others = [vals[o] for o in hs if o != h]
        yo = np.concatenate([o.y for o in others]); Xo = np.concatenate([o.X for o in others])
        p_c = ad.predict_proba(central, v.X).argmax(1)
        results[-1].update(central=macro_f1(v.y, p_c, k), central_acc=float((p_c == v.y).mean()),
                           own_on_others=macro_f1(yo, ad.predict_proba(local[h], Xo).argmax(1), k),
                           glob_on_others=macro_f1(yo, ad.predict_proba(g, Xo).argmax(1), k))
    y = np.concatenate(ys)
    results.append(dict(seed=seed, hospital='ALL', train=sum(trains.values()), held_out=len(y),
                        own=macro_f1(y, np.concatenate(pl), k), glob=macro_f1(y, np.concatenate(pg), k),
                        own_acc=float((np.concatenate(pl) == y).mean()), glob_acc=float((np.concatenate(pg) == y).mean()),
                        central=macro_f1(y, ad.predict_proba(central, np.concatenate([vals[h].X for h in hs])).argmax(1), k)))
    json.dump(dict(modality=modality, rounds=rounds, cap=CAP, results=results), open(out, 'w'), indent=1)
print('done', modality)
