#!/usr/bin/env python3


import pandas as pd
import numpy as np


if __name__ == "__main__":

    feature_names   = np.load("features.npy", allow_pickle=True)
    means           = np.load("means.npy", allow_pickle=True)
    stds            = np.load("stds.npy", allow_pickle=True)
    orderings       = np.load("orderings.npy", allow_pickle=True)

    feature_names   = feature_names
    means           = means[orderings]
    stds            = stds[orderings]


    with open("EXCEL.csv", mode="w+") as f:
        f.write(f"Feature_Name,avg,std\n")
        for name, avg, std in zip(feature_names, means, stds):
            f.write(f"{name},{avg},{std}\n")



            