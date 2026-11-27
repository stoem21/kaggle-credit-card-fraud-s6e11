"""MLP for the blend: different inductive bias from the tree models.

Inputs per CV fold (all label-dependent parts fitted on the training fold only):
- logit of in-fold target encodings,
- numeric columns after a quantile (rank-gauss) transform fitted on the training fold,
- boolean flags as 0/1,
- learned embeddings of the categorical columns.
Early stopping on the same 10% slice of the training fold used by the tree models.
"""
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer

from .config import N_FOLDS
from .data import CATEGORICAL
from .features import target_encode, te_keys
from .metrics import auc

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class PLE(torch.nn.Module):
    """Piecewise-linear encoding of numeric columns (Gorishniy et al. 2022): column j, bin t ->
    clip((x - lo_jt) / width_jt, 0, 1). Bin edges = training-fold quantiles. The first n_num dense columns
    must be the raw numerics to encode. Padded bins have lo = +1e9, so they encode to 0."""
    def __init__(self, lo, width):
        super().__init__()
        self.register_buffer("lo", lo)
        self.register_buffer("width", width)

    def forward(self, x):
        return torch.clamp((x[:, :, None] - self.lo) / self.width, 0, 1).flatten(1)


def ple_edges(Z, n_bins):
    """Z: (rows, cols) numpy, training fold only. Returns lo, width tensors (cols, n_bins)."""
    lo = np.full((Z.shape[1], n_bins), 1e9, dtype=np.float32)
    width = np.ones((Z.shape[1], n_bins), dtype=np.float32)
    for j in range(Z.shape[1]):
        e = np.unique(np.quantile(Z[:, j], np.linspace(0, 1, n_bins + 1)))
        lo[j, :len(e) - 1], width[j, :len(e) - 1] = e[:-1], np.diff(e)
    return torch.tensor(lo), torch.tensor(width)


class MLP(torch.nn.Module):
    def __init__(self, n_dense, cat_sizes, emb_dim=6, hidden=(256, 128), dropout=0.2, ple=None):
        super().__init__()
        self.ple = ple
        self.embs = torch.nn.ModuleList([torch.nn.Embedding(n, emb_dim) for n in cat_sizes])
        n_ple = 0 if ple is None else ple.lo.numel()
        n_raw = 0 if ple is None else ple.lo.shape[0]
        self.n_raw = n_raw
        layers, d = [], n_dense - n_raw + n_ple + emb_dim * len(cat_sizes)
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.SiLU(), torch.nn.Dropout(dropout)]
            d = h
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, dense, cats):
        e = [emb(cats[:, i]) for i, emb in enumerate(self.embs)]
        if self.ple is not None:
            dense = torch.cat([self.ple(dense[:, :self.n_raw]), dense[:, self.n_raw:]], dim=1)
        return self.net(torch.cat([dense] + e, dim=1)).squeeze(1)


def _predict(model, dense, cats, bs=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(dense), bs):
            out.append(torch.sigmoid(model(dense[i:i + bs], cats[i:i + bs])).cpu().numpy())
    return np.concatenate(out)


def _train(dense_fit, cats_fit, y_fit, dense_es, cats_es, y_es, cat_sizes, seed, epochs=30, bs=2048,
           lr=2e-3, wd=1e-5, patience=4, ple=None):
    torch.manual_seed(seed)
    model = MLP(dense_fit.shape[1], cat_sizes, ple=ple).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = torch.nn.BCEWithLogitsLoss()
    yt = torch.tensor(y_fit, dtype=torch.float32, device=DEVICE)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), generator=g).to(DEVICE)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(dense_fit[idx], cats_fit[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        a = auc(y_es, _predict(model, dense_es, cats_es))
        if a > best:
            best, bad, best_ep = a, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep + 1


def run_cv_nn(X, y, X_test, num_cols, bool_cols, te_cols, fold_seed, n_bags=1, smoothing=20, log=print,
              ple_bins=0, train_mask=None, **train_kw):
    """X/X_test: model frames incl. derived columns. Returns dict(auc, oof, test, epochs).
    ple_bins > 0: also feed piecewise-linear encodings of num_cols (edges from training-fold quantiles).
    train_mask: optional boolean array over X rows; only training-fold rows with True are used for TE, scaling,
    fitting and early stopping (a specialist model). Every validation row is still predicted."""
    y = np.asarray(y)
    oof, test_pred, epochs = np.zeros(len(y)), np.zeros(len(X_test)), []
    cat_sizes = [len(X[c].cat.categories) for c in CATEGORICAL]
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=fold_seed)
    for k, (tr, va) in enumerate(folds.split(X, y)):
        t0 = time.time()
        if train_mask is not None:
            tr = tr[np.asarray(train_mask)[tr]]
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        te_tr, te_va, te_te = [], [], []
        for c in te_cols:
            a, (b, t) = target_encode(te_keys(Xtr, c), y[tr], [te_keys(Xva, c), te_keys(X_test, c)],
                                      smoothing=smoothing, seed=fold_seed)
            te_tr.append(a), te_va.append(b), te_te.append(t)
        logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                                 random_state=0).fit(Xtr[num_cols])

        def dense(Z, te):
            parts = [qt.transform(Z[num_cols]), Z[bool_cols].to_numpy(float), logit(np.column_stack(te))]
            if ple_bins:
                parts = [Z[num_cols].to_numpy(float)] + parts
            return torch.tensor(np.hstack(parts), dtype=torch.float32, device=DEVICE)

        def cats(Z):
            return torch.tensor(np.column_stack([Z[c].cat.codes.to_numpy() for c in CATEGORICAL]),
                                dtype=torch.long, device=DEVICE)

        d_tr, c_tr = dense(Xtr, te_tr), cats(Xtr)
        d_va, c_va, d_te, c_te = dense(Xva, te_va), cats(Xva), dense(X_test, te_te), cats(X_test)
        # standardize TE logits etc. with training-fold statistics
        mu, sd = d_tr.mean(0), d_tr.std(0) + 1e-6
        ple = None
        if ple_bins:  # raw numerics stay unscaled (PLE edges are in raw units)
            n_raw = len(num_cols)
            mu[:n_raw], sd[:n_raw] = 0, 1
            ple = PLE(*ple_edges(Xtr[num_cols].to_numpy(float), ple_bins))
        d_tr, d_va, d_te = (d_tr - mu) / sd, (d_va - mu) / sd, (d_te - mu) / sd
        fit_idx, es_idx = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr],
                                           random_state=fold_seed)
        fi, ei = torch.tensor(fit_idx, device=DEVICE), torch.tensor(es_idx, device=DEVICE)
        pv, pt = np.zeros(len(va)), np.zeros(len(X_test))
        for b in range(n_bags):
            m, ep = _train(d_tr[fi], c_tr[fi], y[tr][fit_idx], d_tr[ei], c_tr[ei], y[tr][es_idx], cat_sizes,
                           seed=1000 * b + fold_seed, ple=ple, **train_kw)
            pv += _predict(m, d_va, c_va) / n_bags
            pt += _predict(m, d_te, c_te) / n_bags
            epochs.append(ep)
        oof[va] = pv
        test_pred += pt / N_FOLDS
        log(f"  fold {k}: auc={auc(y[va], pv):.5f} epochs={epochs[-n_bags:]} {time.time() - t0:.0f}s")
    return dict(auc=auc(y, oof), oof=oof, test=test_pred, epochs=epochs)
