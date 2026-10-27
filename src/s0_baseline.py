# Safety baseline: starter features minus telescope-identifying meta, 5-fold OOF.
import numpy as np, pandas as pd, warnings, json, time
warnings.filterwarnings('ignore', category=RuntimeWarning)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss

Xtr = np.load('train_cadences.npy', mmap_mode='r'); Xte = np.load('test_cadences.npy', mmap_mode='r')
tr = pd.read_csv('train_meta.csv'); te = pd.read_csv('test_meta.csv')
LABELS = sorted(tr.label.unique()); y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values

def features(X, batch=2000):
    out = []
    for s in range(0, len(X), batch):
        x = X[s:s + batch].astype(np.float32); x[x == 0] = np.nan
        x = x - np.nanmedian(x, axis=(2, 3), keepdims=True)
        prof = np.nanmean(x, axis=2); f = {}
        for p in range(6):
            f[f'p{p}_std'] = np.nanstd(x[:, p], axis=(1, 2)); f[f'p{p}_max'] = np.nanmax(x[:, p], axis=(1, 2))
            f[f'p{p}_peak'] = np.nanmax(prof[:, p], axis=1)
            f[f'p{p}_rowmax'] = np.nanmax(np.nanmean(x[:, p], axis=2), axis=1)
        df = pd.DataFrame(f)
        for k in ['std', 'max', 'peak', 'rowmax']:
            df[f'onoff_{k}'] = df[[f'p{p}_{k}' for p in (0, 2, 4)]].mean(1) - df[[f'p{p}_{k}' for p in (1, 3, 5)]].mean(1)
        out.append(df)
    return pd.concat(out, ignore_index=True)

t = time.time(); Ftr = features(Xtr); Fte = features(Xte); print('feats', round(time.time() - t, 1), 's')
if __name__ == '__main__':
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Ftr, y)): folds[va] = k
    np.save('oof/folds.npy', folds)
    oof = np.zeros((len(y), 9)); tst = np.zeros((len(te), 9))
    for k in range(5):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, class_weight='balanced', random_state=0)
        m.fit(Ftr[folds != k], y[folds != k]); oof[folds == k] = m.predict_proba(Ftr[folds == k]); tst += m.predict_proba(Fte) / 5
    np.save('oof/s0_hgb__oof.npy', oof); np.save('oof/s0_hgb__test.npy', tst)
    pred = oof.argmax(1)
    print('OOF macroF1', round(f1_score(y, pred, average='macro'), 4), 'logloss', round(log_loss(y, oof), 4))
    print(pd.Series(f1_score(y, pred, average=None), index=LABELS).round(3).to_string())
    sub = pd.DataFrame({'id': te.id, 'label': np.array(LABELS)[tst.argmax(1)]}); sub.to_csv('subs/s0_hgb.csv', index=False)
    print(sub.label.value_counts().to_string())
