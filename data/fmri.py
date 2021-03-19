import os
import numpy as np
from tqdm import tqdm

_DPATH = "/nfs/gatsbystor/ilyesk/ds000224/transformed"


def zpad(idx):
    if len(str(idx)) < 2:
        return f'0{idx}'
    return f'{idx}'


def load_single(path=_DPATH, sub_idx=1, ses_idx=1, dtype=np.float32):
    return np.loadtxt(os.path.join(path,
                                   f'sub{zpad(sub_idx)}',
                                   f'sub{zpad(sub_idx)}_ses{zpad(ses_idx)}.txt'),
                      dtype=dtype).T


def load_subject(path=_DPATH, sub_idx=1, ses_idx='all', dtype=np.float32):
    if not isinstance(sub_idx, int):
        raise ValueError(
            f"illegal value in {sub_idx}; use `load` to load more than one subject")
    ses_idx = _check_idx(ses_idx)
    X = []
    print(f"Loading sessions {ses_idx}...")
    for ses in tqdm(ses_idx):
        X.append(load_single(path, sub_idx=sub_idx, ses_idx=ses, dtype=dtype))
    return X


def load_session(path=_DPATH, sub_idx='all', ses_idx=1, dtype=np.float32):
    if not isinstance(ses_idx, int):
        raise ValueError(
            f"illegal value in {ses_idx}; use `load` to load more than one session")
    sub_idx = _check_idx(sub_idx)
    X = []
    print(f"Loading subjects {sub_idx}...")
    for sub in tqdm(sub_idx):
        X.append(load_single(path, sub_idx=sub, ses_idx=ses_idx, dtype=dtype))
    return X


def load(path=_DPATH, sub_idx=1, ses_idx=1, sub_ses_map=None, dtype=np.float32):
    if sub_ses_map is not None:
        sub_ses_map = _check_sub_ses_map(sub_ses_map)
        use_ssm = True
    else:
        sub_idx = _check_idx(sub_idx)
        ses_idx = _check_idx(ses_idx)
        use_ssm = False
    X = []
    if use_ssm:
        for sub, ses in sub_ses_map.items():
            print(f"Loading data for subject {sub}..")
            X.extend(load_subject(path=path, sub_idx=sub,
                                  ses_idx=ses, dtype=dtype))
    else:
        for sub in sub_idx:
            print(f"Loading data for subject {sub}..")
            X.extend(load_subject(path=path, sub_idx=sub,
                                  ses_idx=ses_idx, dtype=dtype))
    return X


def _check_idx(idx):
    if isinstance(idx, str):
        if idx == 'all':
            idx = np.arange(1, 11)
        elif idx == '1h' or idx == 'first_half':
            idx = np.arange(1, 6)
        elif idx == '2h' or idx == 'second_half':
            idx = np.arange(6, 11)
        else:
            try:
                idx = [int(idx)]
            except ValueError:
                raise ValueError(f"illegal value in ses_idx: {idx}")
    elif isinstance(idx, int):
        idx = [idx]
    elif isinstance(idx, (tuple, set, list, range, np.ndarray)):
        pass
    else:
        raise ValueError(f"illegal value in ses_idx: {idx}")
    return idx


def _check_sub_ses_map(ssm):
    if not isinstance(ssm, dict):
        raise ValueError("sub_ses_map should be a dict")
    valid_keys = [zpad(idx) for idx in range(1, 11)]
    for k, v in ssm.items():
        if zpad(k) not in valid_keys:
            raise ValueError(f"illegal key in sub_ses_map: {k}")
        ssm[k] = _check_idx(v)
    return ssm
