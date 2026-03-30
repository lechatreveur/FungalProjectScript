#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:28:26 2025

@author: user
"""

import os, pandas as pd
from datetime import datetime

def path(*parts): return os.path.join(*parts)

def load_csv(*parts, **read_csv_kwargs):
    return pd.read_csv(path(*parts), **read_csv_kwargs)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True); return d

def timestamp(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)

def save_csv(df, *parts, index=False):
    p = path(*parts); ensure_dir(os.path.dirname(p)); df.to_csv(p, index=index); return p
