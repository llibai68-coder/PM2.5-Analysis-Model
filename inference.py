import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import features as ft

_bundle = None

NON_NEGATIVE = {'WSPM', 'NO2', 'CO', 'O3'}

FACTOR_TEXT = {
    'TEMP': ('ອຸນຫະພູມສູງກວ່າປົກກະຕິ', 'ອຸນຫະພູມຕ່ຳກວ່າປົກກະຕິ ອາດກ່ຽວກັບການເຜົາໄໝ້ເພື່ອໃຫ້ຄວາມອົບອຸ່ນ'),
    'PRES': ('ຄວາມກົດອາກາດສູງ ອາກາດນິ່ງ ຝຸ່ນສະສົມງ່າຍ', 'ຄວາມກົດອາກາດຕ່ຳກວ່າປົກກະຕິ ອາກາດເຄື່ອນໄຫວດີກວ່າ'),
    'DEWP': ('ຄວາມຊື້ນສູງ ອະນຸພາກຮວມຕົວ/ດູດຊຶມນ້ຳໄດ້ງ່າຍ', 'ອາກາດແຫ້ງກວ່າປົກກະຕິ'),
    'WSPM': ('ລົມແຮງກວ່າປົກກະຕິ ຊ່ວຍກະຈາຍຝຸ່ນ', 'ລົມອ່ອນກວ່າປົກກະຕິ ຝຸ່ນບໍ່ຖືກພັດໄປ ຈຶ່ງສະສົມ'),
    'NO2':  ('NO2 ສູງ ສັນຍານຈາກຈະລາຈອນ/ການເຜົາໄໝ້', 'NO2 ຕ່ຳກວ່າປົກກະຕິ'),
    'CO':   ('CO ສູງ ສັນຍານການເຜົາໄໝ້ບໍ່ສົມບູນ (ຖ່ານຫີນ/ເຄື່ອງຈັກ)', 'CO ຕ່ຳກວ່າປົກກະຕິ'),
    'O3':   ('O3 ສູງ ແສງແດດແຮງ ປະຕິກິລິຍາ photochemical', 'O3 ຕ່ຳກວ່າປົກກະຕິ ມັກພົບຕອນມົນລະພິດໜັກ'),
}


def _predict_many(model, X):
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(X))
    return model.predict(X)


def _check_features(col, model):
    names = None
    if isinstance(model, xgb.Booster):
        names = model.feature_names
    elif hasattr(model, 'feature_names_in_'):
        names = list(model.feature_names_in_)
    if names and list(names) != list(ft.FEATURES_INPUT):
        raise RuntimeError(
            f"ໂມເດວ {col} ເທຣນດ້ວຍຟີເຈີ {list(names)} "
            f"ແຕ່ features.py ໃຊ້ {list(ft.FEATURES_INPUT)}. "
            "ຕ້ອງໃຊ້ features.py ສະບັບດຽວກັນກັບທີ່ເທຣນໂມເດວ."
        )


def _load_boosters(base_dir, cols):
    models = {}
    for col in cols:
        f = os.path.join(base_dir, f'{col}_model.json')
        if not os.path.exists(f):
            raise FileNotFoundError(f'ບໍ່ພົບໄຟລ໌ໂມເດວ: {f}')
        bst = xgb.Booster()
        bst.load_model(f)
        models[col] = bst
    return models


def load(path=None):
    global _bundle
    if _bundle is not None:
        return _bundle

    here = os.path.dirname(os.path.abspath(__file__))
    path = path or os.path.join(here, 'weather_causes_model.pkl')
    base_dir = os.path.dirname(os.path.abspath(path))

    b = dict(joblib.load(path))
    b['target_cols'] = list(ft.TARGET_COLS)
    b.setdefault('feature_cols', list(ft.FEATURES_INPUT))
    meta = dict(b.get('meta') or {})
    meta.setdefault('station', 'Changping')
    b['meta'] = meta

    if 'models' not in b:
        b['models'] = _load_boosters(base_dir, b['target_cols'])
    for col in b['target_cols']:
        _check_features(col, b['models'][col])

    missing = [k for k in ('base_mean', 'base_std', 'resid_q', 'test_r2') if k not in b]
    if missing:
        cols = b['target_cols']
        tr = pd.read_csv(os.path.join(base_dir, 'N_train.csv'), index_col=0, parse_dates=True)
        te = pd.read_csv(os.path.join(base_dir, 'N_test.csv'), index_col=0, parse_dates=True)
        if 'base_mean' in missing:
            keys = [tr.index.month.rename('month'), tr.index.hour.rename('hour')]
            b['base_mean'] = tr.groupby(keys)[cols].mean()
        if 'base_std' in missing:
           keys = [tr.index.month.rename('month'), tr.index.hour.rename('hour')]
           b['base_std'] = tr.groupby(keys)[cols].std()
        meta.setdefault('train_end', str(tr.index.max()))
        if 'resid_q' in missing or 'test_r2' in missing:
            Xte = te[ft.FEATURES_INPUT]
            rq, r2 = {}, {}
            for col in cols:
                y = te[col].to_numpy()
                resid = y - _predict_many(b['models'][col], Xte)
                rq[col] = (float(np.quantile(resid, 0.10)), float(np.quantile(resid, 0.90)))
                r2[col] = float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
            b.setdefault('resid_q', rq)
            b.setdefault('test_r2', r2)

    _bundle = b
    return _bundle


def predict_causes(pm25, hour, month):
    if not (0 <= pm25 <= 1000):
        raise ValueError('PM2.5 ຕ້ອງຢູ່ລະຫວ່າງ 0 - 1000')
    if not (0 <= hour <= 23):
        raise ValueError('hour ຕ້ອງຢູ່ລະຫວ່າງ 0 - 23')
    if not (1 <= month <= 12):
        raise ValueError('month ຕ້ອງຢູ່ລະຫວ່າງ 1 - 12')

    b = load()
    X = ft.make_X(pm25, hour, month)

    out = {}
    for col in b['target_cols']:
        v = float(_predict_many(b['models'][col], X)[0])
        lo, hi = b['resid_q'][col]
        low, high = v + lo, v + hi
        if col in NON_NEGATIVE:
            v, low, high = max(v, 0), max(low, 0), max(high, 0)
        out[col] = {'pred': v, 'low': low, 'high': high}
    return out


def explain_causes(pm25, hour, month):
    b = load()
    pred = predict_causes(pm25, hour, month)
    rows = []
    for col, d in pred.items():
        # 💡 ຈຸດນີ້ສຳຄັນ: ຕ້ອງສ້າງ key ເປັນ (month, hour) ຖ້າ base_mean ເປັນ MultiIndex
        key = (month, hour) if isinstance(b['base_mean'].index, pd.MultiIndex) else month
        
        mean = float(b['base_mean'].loc[key, col])
        std = float(b['base_std'].loc[key, col])
        z = (d['pred'] - mean) / std if std > 0 else 0.0
        
        level = 'ປົກກະຕິ' if abs(z) < 0.5 else ('ຜິດປົກກະຕິເລັກນ້ອຍ' if abs(z) < 1 else 'ຜິດປົກກະຕິຫຼາຍ')
        rows.append({
            'ປັດໄຈ': col,
            'ຄ່າທຳນາຍ': round(d['pred'], 2),
            'ຊ່ວງ (P10-P90)': f"{d['low']:.1f} - {d['high']:.1f}",
            'ຄ່າປົກກະຕິຂອງເດືອນ': round(mean, 2),
            'z-score': round(z, 2),
            'ລະດັບ': level,
            'ຄວາມໝາຍ': FACTOR_TEXT[col][0 if z > 0 else 1] if abs(z) >= 0.5 else '-',
            'ຄວາມເຊື່ອຖືໂມເດວ (R²)': round(b['test_r2'][col], 2),
        })
    table = pd.DataFrame(rows)
    table = table.reindex(table['z-score'].abs().sort_values(ascending=False).index)
    return table.reset_index(drop=True)
