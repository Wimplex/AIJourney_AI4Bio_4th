import random
from typing import Dict, Any, Iterable
from itertools import product

import numpy as np
import pandas as pd

from constants import TARGET_LABEL


class FrequencyShift:
    def __init__(self, shift: float = 0.5):
        self.shift = shift

    def __call__(self, class_df: pd.DataFrame, by: str = "spec"):
        new_df = class_df.copy()
        new_df[by] = new_df[by].apply(lambda x: np.roll(x, int(self.shift * len(x))))
        new_class_name = class_df[TARGET_LABEL].unique()[0] + f"_shifted_by_{self.shift}"
        new_df.loc[:, TARGET_LABEL] = new_class_name
        return new_df


class FrequencyInversion:
    def __call__(self, class_df: pd.DataFrame, by: str = "spec"):
        new_df = class_df.copy()
        new_df[by] = new_df[by].apply(lambda x: np.flip(x))
        new_class_name = class_df[TARGET_LABEL].unique()[0] + "_freq_inv"
        new_df.loc[:, TARGET_LABEL] = new_class_name
        return new_df


class IntensivityReflection:
    def __init__(self, reflection_threshold: float = 0.5):
        self.refl_thr = reflection_threshold

    def __call__(self, class_df: pd.DataFrame, class_suffix: str = None, by: str = "spec"):

        def apply_func(x):
            x = 2 * self.refl_thr - x
            x[x < 0.0] = 0.0
            return x

        new_df = class_df.copy()
        new_df[by] = new_df[by].apply(apply_func)

        suffix = f"intens_refl_on_{self.refl_thr}" if class_suffix is None else class_suffix
        new_class_name = class_df[TARGET_LABEL].unique()[0] + "_" + suffix
        new_df.loc[:, TARGET_LABEL] = new_class_name
        return new_df


class IntensivityInversion:
    def __call__(self, class_df: pd.DataFrame, by: str = "spec"):
        return IntensivityReflection()(class_df, "int_inv", by)


gen_presets = [
    IntensivityInversion(),
    FrequencyInversion(),
    FrequencyShift(0.25),
    FrequencyShift(0.5),
    FrequencyShift(0.75),
]


def generate_data(df: pd.DataFrame, feats_names: Iterable[str], generation_params: Dict[str, Any]):
    """ Generates new classes """

    if generation_params["num_new_classes"] == 0:
        return pd.DataFrame()

    targets = df[TARGET_LABEL].unique()
    generations = list(product(targets, gen_presets))
    random.shuffle(generations)
    generations = generations[:generation_params["num_new_classes"]]

    gen_df = pd.DataFrame()
    for tar, gen in generations:
        curr_df = df[df[TARGET_LABEL] == tar]
        for feat in feats_names:
            curr_df = gen(curr_df, by=feat)
        gen_df = pd.concat([gen_df, curr_df])
    return gen_df