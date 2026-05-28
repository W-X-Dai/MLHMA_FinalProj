import os
import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset

class GaitDataset(Dataset):
    def __init__(self, base_path, target_len=100):
        self.data = []
        self.target_len = target_len
        self.g = 9.81
        
        pt_list = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        
        for pt in pt_list:
            pt_path = os.path.join(base_path, pt)
            pt_samples = []
            has_nan    = False

            for trial in sorted(os.listdir(pt_path)):
                if not trial.endswith('.mat'):
                    continue
                mat = scipy.io.loadmat(os.path.join(pt_path, trial))

                bw_newton   = mat['BodyWeight'][0][0] * self.g
                grf         = mat['GRF'] / bw_newton
                com_cop_rel = (mat['COM'][:, :2] - mat['COP']) / 1000.0

                if np.isnan(grf).any() or np.isnan(com_cop_rel).any():
                    print(f"[SKIP patient] {pt} — NaN in {trial}, dropping all {pt} trials")
                    has_nan = True
                    break

                # CoM height is ~57% of standing height (Dempster 1955)
                height_m = float(mat['COM'][:, 2].mean() / 0.57 / 1000.0)

                pt_samples.append({
                    'patient_id': pt,
                    'height_m':   height_m,
                    'input':  torch.FloatTensor(self._interpolate(grf, target_len)),
                    'target': torch.FloatTensor(self._interpolate(com_cop_rel, target_len)),
                })

            if not has_nan:
                self.data.extend(pt_samples)

    def _interpolate(self, data, length):
        x_old = np.linspace(0, 1, data.shape[0])
        x_new = np.linspace(0, 1, length)

        return np.array([np.interp(x_new, x_old, data[:, i]) for i in range(data.shape[1])]).T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['target']