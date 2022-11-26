from typing import Dict, Any

import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from constants import TARGET_LABEL


class Normalize:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, spectrum: np.ndarray):
        if np.random.uniform() > self.p:
            return spectrum
        return normalize(np.expand_dims(spectrum, axis=0)).squeeze()


class PeakElimination:
    """ Randomly sets some fraction of peaks to zero """
    def __init__(
        self, 
        min_peaks_frac: float, 
        max_peaks_frac: float, 
        inpute_val: float = 0.0, 
        p: float = 0.5
    ):
        self.min_ = min_peaks_frac
        self.max_ = max_peaks_frac
        self.p = p
        self.inpute_val = inpute_val

    def __call__(self, spectrum: np.ndarray):
        if np.random.uniform() > self.p:
            return spectrum

        num_feats = len(spectrum)
        frac = np.random.uniform(self.min_, self.max_)
        random_ids = np.random.choice(
            np.arange(0, num_feats), int(frac * num_feats))

        if self.inpute_val != "random":
            spectrum[random_ids] = self.inpute_val
        else:
            rand = np.random.uniform(0.0, 1.2, size=len(random_ids))
            rand = rand * np.max(spectrum)
            spectrum[random_ids] = rand
            
        return spectrum


class SpectrumShift:
    """ Synthesize in-class instance by variating per-peak values on in-class std """

    def __init__(self, shift_std: float = 0.005, p: float = 0.5):
        self.shift_std = shift_std
        self.p = p

    def __call__(self, spectrum: np.ndarray):
        if np.random.uniform() > self.p:
            return spectrum

        shift = np.random.normal(0.0, self.shift_std)
        spectrum += shift
        spectrum[spectrum < 0.0] = 0.0
        return spectrum


class VarianceFiltration:
    """ Nullifies features with small variance """

    def __init__(self, std_threshold: float = 0.01, p: float = 0.5):
        self.std_thr = std_threshold
        self.p = p

    def __call__(self, spectrum: np.ndarray, per_peak_std: np.ndarray):
        if np.random.uniform() > self.p:
            return spectrum 

        spectrum[per_peak_std < self.std_thr] = 0.0
        return spectrum


class PeakVariation:
    """ Synthesize in-class instance by variating per-peak values on in-class stds """

    def __init__(self, fraction_of_peaks: float = 0.5, std_factor: float = 0.3, p: float = 0.5):
        self.fraction_of_peaks = fraction_of_peaks
        self.std_factor = std_factor
        self.p = p

    def __call__(self, spectrum: np.ndarray, per_peak_std: np.ndarray):
        if np.random.uniform() > self.p:
            return spectrum

        indent = np.array([np.random.normal(0.0, self.std_factor * sigma) for sigma in per_peak_std])
        random_repl_ids = np.random.randint(0, len(indent), int(len(indent) * self.fraction_of_peaks))
        spectrum[random_repl_ids] += indent[random_repl_ids]
        spectrum[spectrum < 0.0] = 0.0
        return spectrum


class NoisePerturbation:
    """ Creates new in-class instance by adding some small amount of Gaussian noise """

    def __init__(self, noise_std: float = 0.001, p: float = 0.5):
        self.noise_std = noise_std
        self.p = p
        
    def __call__(self, spectrum: np.ndarray):
        if np.random.uniform() > self.p:
            return spectrum

        noise = np.random.normal(0.0, self.noise_std, len(spectrum))
        spectrum += noise
        spectrum[spectrum < 0.0] = 0.0
        return spectrum


class SegmentsCombination:
    """ Creates new instances of each class by combining spectrum fragments of each member """

    def __init__(self, num_segments: int = 8, p: float = 0.5):
        self.num_segments = num_segments
        self.p = p

    def __call__(self, in_class_instances: np.ndarray):
        if np.random.uniform() > self.p:
            return None

        n_features = in_class_instances[0].shape[0]
        bounds = [0] + np.sort(np.random.randint(0, n_features, size=self.num_segments - 1)).tolist() + [None]
        class_instance_ids = np.random.choice(np.arange(len(in_class_instances)), size=self.num_segments)
        segments = [
            in_class_instances[inst_id][l:r] 
            for inst_id, l, r in zip(class_instance_ids, bounds[:-1], bounds[1:])
        ]
        spectrum = np.concatenate(segments)
        return spectrum


class MixUp:
    def __init__(self, ratio: float = 0.5, p: float = 0.5):
        self.ratio = ratio
        self.p = p

    def __call__(self, x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray):
        if np.random.uniform() > self.p:
            return x1, y1

        new_x = self.ratio * x1 + (1.0 - self.ratio) * x2
        new_y = self.ratio * y1 + (1.0 - self.ratio) * y2
        return new_x, new_y
            

def augment_data(
    df: pd.DataFrame, 
    feats_names: str,
    augmentation_params: Dict[str, Any],
):
    """ Applies in-class augmentations and synthesis tricks for each class """

    gen_df = pd.DataFrame()
    num_augs = augmentation_params["num_augs_per_class"]
    pbar = tqdm.tqdm(
        total=df[TARGET_LABEL].unique().shape[0] * num_augs, 
        desc="Augmentation"
    )
    for tar in df[TARGET_LABEL].unique():
        class_df = df[df[TARGET_LABEL] == tar]

        for _ in range(num_augs):
            feat2spec = {f: None for f in feats_names}
            spec_df = pd.DataFrame(dtype="object")
            spec_df = spec_df.assign(**feat2spec)
            
            for feat in feats_names:
                spectrum = class_df.sample(1)[feat].values[0].copy()
                for aug in augmentation_params["blocks"][feat]:
                    if isinstance(aug, SegmentsCombination):
                        result = aug(class_df[feat].values)
                        spectrum = result if result is not None else spectrum

                    elif isinstance(aug, PeakVariation) or isinstance(aug, VarianceFiltration):
                        per_peak_std = class_df[feat].values.std()
                        spectrum = aug(spectrum, per_peak_std)

                    else:
                        spectrum = aug(spectrum)

                feat2spec[feat] = spectrum

            for feat, spec in feat2spec.items():
                spec_df.loc[0, feat] = spec.reshape(-1, 1)
            spec_df.loc[0, TARGET_LABEL] = tar
            gen_df = pd.concat([gen_df, spec_df])
            pbar.update(1)

    pbar.close()
    return gen_df
