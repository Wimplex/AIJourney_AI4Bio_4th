import numpy as np
import pandas as pd
from scipy.io import loadmat


def glf(
    mz: np.ndarray, 
    intens: np.ndarray, 
    delta_cts: float = 0.1, 
    delta_mz: float = 2000.0, 
    k: float = 7.0
) -> np.ndarray:
    y1 = np.max(intens) * 0.02
    y2 = (np.max(intens) - y1) * delta_cts
    res = y1 + y2 / (1 + np.e ** (y2 * k / (mz - delta_mz)))
    return res


def parse_database(database: np.ndarray, strain_precision = "full") -> pd.DataFrame:

    prec = {"genus": 1, "species": 2, "full": 3}

    num_entities = len(database)
    mz, peaks, strain = [], [], []
    for i in range(num_entities):
        curr_entity = database[i][0]
        for obs_id in range(len(curr_entity["tax"])):
            observation = curr_entity[obs_id]
            mz.append(observation["pik"][0])
            peaks.append(observation["pik"][1])
            st = observation["tax"][0]
            st = st.split(" ")
            st = " ".join(st[:prec[strain_precision]])
            strain.append(st)

    df = pd.DataFrame(columns=["Intens.", "m/z", "strain"], dtype="object")
    df["Intens."] = peaks
    df["m/z"] = mz
    df["strain"] = strain
    return df


def read_pkf(pkf_path: str, strain_precision: str = "full"):
    data = loadmat(pkf_path)
    database = data["C"]["dbs"][0]
    df = parse_database(database, strain_precision)
    return df
