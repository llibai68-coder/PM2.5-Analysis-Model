import pandas as pd
import numpy as np

WEATHER = ['TEMP', 'PRES', 'DEWP', 'WSPM']
GASES   = ['NO2', 'CO', 'O3']
TARGET_COLS = WEATHER + GASES

TIME_COLS = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
FEATURES_INPUT = ['PM2.5'] + TIME_COLS

def _add_time(d):
    d = d.copy()
    d['hour_sin']  = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos']  = np.cos(2 * np.pi * d['hour'] / 24)
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    return d

def build_raw_inverse(df):
    raw = pd.DataFrame(index=df.index)
    raw['PM2.5'] = df['PM2.5']
    raw['hour']  = df.index.hour
    raw['month'] = df.index.month
    for col in TARGET_COLS:
        raw[col] = df[col]
    return _add_time(raw)

def make_X(pm25, hour, month):
    d = pd.DataFrame([{'PM2.5': pm25, 'hour': hour, 'month': month}])
    return _add_time(d)[FEATURES_INPUT]
