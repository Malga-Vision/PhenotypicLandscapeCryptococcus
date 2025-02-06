#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import confusion_matrix
plt.rcParams['figure.dpi'] = 1000  # Figure DPI
plt.rcParams['axes.grid'] = False  # Show grid
plt.rcParams['axes.titlesize'] = 'large'  # Title font size
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.labelsize'] = 'medium'  # Label font size
plt.rcParams['xtick.labelsize'] = 'small'  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 'small'  # Y-axis tick font size
plt.rcParams['legend.fontsize'] = 'medium'  # Legend font size
plt.rcParams['lines.linewidth'] = 6.0  # Line width
plt.rcParams['lines.markersize'] = 1  # Marker size

# Set color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# Set fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Dejavu Sans'] + plt.rcParams['font.serif']
plt.rcParams["font.size"]  = 20
plt.rcParams["font.weight"]  = "bold"

# Set LaTeX support
plt.rcParams['text.usetex'] = False

# Set other parameters
plt.rcParams['savefig.format'] = 'svg' 

def load_arrays():
    features    = np.load("features.npy",allow_pickle=True)
    sensitivity = np.load("sensitivity_all.npy",allow_pickle=True)
    specificity = np.load("specificity_all.npy",allow_pickle=True)
    y_pred_cumulative = np.load("y_pred_cumulative.npy")
    y_test_cumulative = np.load("y_test_cumulative.npy")
    dummy_preds_prior_cumulative = np.load("dummy_preds_prior_cumulative.npy", allow_pickle=True)
    dummy_preds_uniform_cumulative = np.load("dummy_preds_uniform_cumulative.npy", allow_pickle=True)
    dummy_preds_stratified_cumulative = np.load("dummy_preds_stratified_cumulative.npy", allow_pickle=True)    

    with open("sensitivity_specificity.csv", mode="w+") as f:
        f.write(f"Feature_Name,sensitivity,specificity\n")
        for name, sens, spec in zip(features, sensitivity, specificity):
            f.write(f"{name},{sens},{spec}\n")
    
    print(
        len(features), 
        len(sensitivity), 
        len(specificity),
        len(y_pred_cumulative),
        len(y_test_cumulative),
        len(dummy_preds_prior_cumulative),
        len(dummy_preds_uniform_cumulative),
        len(dummy_preds_stratified_cumulative)
    )

    return (
        features, 
        sensitivity, 
        specificity, 
        y_pred_cumulative,
        y_test_cumulative,
        dummy_preds_prior_cumulative, 
        dummy_preds_uniform_cumulative,
        dummy_preds_stratified_cumulative
    )

def constline_figure():
    plt.figure(figsize=(32, 18))    
    plt.plot(np.arange(len(sensitivity)), sensitivity_smooth, color="red", label="Sensitivity")
    plt.plot(np.arange(len(specificity)), specificity_smooth, color="blue", label="Specificity")
    plt.hlines(uniform_sensitivity,     xmin=0, xmax=len(sensitivity)-1, color="#00ff00", linestyles=(0, (3, 1, 1, 1)), label="Sensitivity of Random Classifier (Uniform priors)")    
    plt.hlines(uniform_specificity,     xmin=0, xmax=len(specificity)-1, color="#00aa00", linestyles=(0, (3, 1, 1, 1)), label="Specificity of Random Classifier (Uniform priors)")
    plt.hlines(stratified_sensitivity,  xmin=0, xmax=len(sensitivity)-1, color="#00bbaa", linestyles=(0, (3, 1, 1, 1)), label="Sensitivity of Random Classifier (Class priors)")
    plt.hlines(stratified_specificity,  xmin=0, xmax=len(specificity)-1, color="#00eeaa", linestyles=(0, (3, 1, 1, 1)), label="Specificity of Random Classifier (Class priors)")

    plt.xlabel("Input Features")

    plt.xticks(ticks=np.arange(len(sensitivity)), labels=features[:len(sensitivity)], rotation=90)
    plt.ylim((0, 1))
    plt.title("Impact of Feature Selection according to Feature Importance")
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig("IMAGES/across_features_nullmodels_constantline.svg", format="svg", dpi=1000)
    plt.close()

def marker_atX0_figure():
    plt.figure(figsize=(32, 18))    
    plt.plot(np.arange(len(sensitivity)), sensitivity_smooth, color="red", label="Random Forest Sensitivity")
    plt.plot(np.arange(len(specificity)), specificity_smooth, color="blue", label="Random Forest Specificity")
    plt.scatter(0, uniform_sensitivity,    color="#00ff00", marker="d", s=400, edgecolors="black", linewidths=2, label="Random Classifier Sensitivity (Uniform priors)")
    plt.scatter(0, stratified_sensitivity, color="#00bbaa", marker="d", s=400, edgecolors="black", linewidths=2, label="Random Classifier Sensitivity (Class priors)")
    plt.scatter(0, uniform_specificity,    color="#00ff00", marker="X", s=400, edgecolors="black", linewidths=2, label="Random Classifier Specificity (Uniform priors)")
    plt.scatter(0, stratified_specificity, color="#00bbaa", marker="X", s=400, edgecolors="black", linewidths=2, label="Random Classifier Specificity (Class priors)")

    plt.xlabel("Input Features")

    plt.xticks(ticks=np.arange(len(sensitivity)), labels=features[:len(sensitivity)], rotation=90)
    plt.ylim((0, 1))
    plt.title("Impact of Feature Selection according to Feature Importance")
    plt.tight_layout()
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig("IMAGES/across_features_nullmodels_markerX0.svg", format="svg", dpi=1000)
    plt.close()

if __name__ == "__main__":
    (
        features, 
        sensitivity, 
        specificity, 
        y_pred_cumulative, 
        y_test_cumulative, 
        dummy_preds_prior_cumulative, 
        dummy_preds_uniform_cumulative,
        dummy_preds_stratified_cumulative,
    ) = load_arrays()

    dummy_prior_conf   = confusion_matrix(dummy_preds_prior_cumulative, y_test_cumulative)
    dummy_uniform_conf = confusion_matrix(dummy_preds_uniform_cumulative, y_test_cumulative)
    dummy_stratified_conf = confusion_matrix(dummy_preds_stratified_cumulative, y_test_cumulative)

    prior_tp = dummy_prior_conf[0, 0]
    prior_fp = dummy_prior_conf[0, 1]
    prior_fn = dummy_prior_conf[1, 0]
    prior_tn = dummy_prior_conf[1, 1]

    prior_sensitivity = prior_tp / (prior_tp + prior_fn)
    prior_specificity = prior_tn / (prior_tn + prior_fp)

    uniform_tp = dummy_uniform_conf[0, 0]
    uniform_fp = dummy_uniform_conf[0, 1]
    uniform_fn = dummy_uniform_conf[1, 0]
    uniform_tn = dummy_uniform_conf[1, 1]

    uniform_sensitivity = uniform_tp / (uniform_tp + uniform_fn)
    uniform_specificity = uniform_tn / (uniform_tn + uniform_fp)

    stratified_tp = dummy_stratified_conf[0, 0]
    stratified_fp = dummy_stratified_conf[0, 1]
    stratified_fn = dummy_stratified_conf[1, 0]
    stratified_tn = dummy_stratified_conf[1, 1]

    stratified_sensitivity = stratified_tp / (stratified_tp + stratified_fn)
    stratified_specificity = stratified_tn / (stratified_tn + stratified_fp)
    window_size = 5  
    sensitivity_smooth = uniform_filter1d(sensitivity, size=window_size)
    specificity_smooth = uniform_filter1d(specificity, size=window_size)

    constline_figure()
    marker_atX0_figure()


    


