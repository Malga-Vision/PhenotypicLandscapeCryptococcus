#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import csv
import random
from scipy import interpolate
# random.seed(1)
from sklearn.metrics import confusion_matrix
import gc
# plt.rcParams.update({'font.size': 14})
import sys
from sklearn.dummy import DummyClassifier

# Set global parameters
# plt.rcParams['figure.figsize'] = (8, 6)  # Figure size
plt.rcParams['figure.dpi'] = 1000  # Figure DPI
plt.rcParams['axes.grid'] = False  # Show grid
plt.rcParams['axes.titlesize'] = 'large'  # Title font size
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.labelsize'] = 'medium'  # Label font size
plt.rcParams['xtick.labelsize'] = 'small'  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 'small'  # Y-axis tick font size
plt.rcParams['legend.fontsize'] = 'medium'  # Legend font size
plt.rcParams['lines.linewidth'] = 4.0  # Line width
plt.rcParams['lines.markersize'] = 1  # Marker size

# Set color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# Set fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Dejavu Sans'] + plt.rcParams['font.serif']
plt.rcParams["font.size"]  = 16
plt.rcParams["font.weight"]  = "bold"

# Set LaTeX support
plt.rcParams['text.usetex'] = False
plt.rcParams['savefig.format'] = 'svg' 

def get_confusion_matrix(confusion_matrix_final,labels):
    for i in range(0,len(np.unique(labels))):
        confusion_matrix_final[i,:] = confusion_matrix_final[i,:] / np.sum(confusion_matrix_final, axis=1)[i]
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    plt.figure(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_final)
    disp.plot(cmap='gray', values_format='.3f')
    disp.ax_.set_xticklabels(labels)
    disp.ax_.set_yticklabels(labels)
    disp.ax_.set_xlabel('True')
    disp.ax_.set_ylabel('Predicted')
    plt.tight_layout()
    plt.savefig("IMAGES/ConfusionMatrix.svg", format="svg", dpi=1000)
    plt.close()

    return confusion_matrix_final

def return_importances_plot(importances_list, stressors,number):    
    plt.figure()
    means = np.asarray(importances_list).mean(axis=0)
    stds = np.asarray(importances_list).std(axis=0)

    ordering = np.argsort(means)[::-1]

    forest_importances = pd.Series(means[ordering], stressors.columns[1:][ordering])
    fig, ax = plt.subplots(figsize=(30,15))

    forest_importances.plot.bar(ax=ax,yerr=stds[ordering])
    r = forest_importances.to_numpy()
    ordering = ordering[0:number]

    features = (stressors.columns[ordering + 1])
    r = r[ordering]
    ax.set_title("Feature Importance  (Mean Decrease in Impurity)")
    ax.set_ylabel("Mean Decrease in Impurity (MDI)")
    fig.tight_layout()
    plt.savefig('./IMAGES/feature_importances.svg', format="svg", dpi=1000)
    plt.close()
    return features, means, stds, ordering


def return_importances(importances_list, stressors,number):
    means = np.asarray(importances_list).mean(axis=0)
    stds = np.asarray(importances_list).std(axis=0)
    forest_importances = pd.Series(means, stressors.columns[1:])

    r = forest_importances.to_numpy()

    ordering = np.argsort(r)[::-1][0:number]

    features = (stressors.columns[ordering + 1])
    return features, r[ordering]

def extract_train(mouse,stressors):

  indexes_for_stressors = list()
  for i in range(0, mouse.shape[0]):
      indexes_for_stressors.append(np.where(stressors.iloc[:,0].eq(mouse.iloc[:,0].values[i]))[0][0])

  x_train= stressors.iloc[indexes_for_stressors,1:].to_numpy()
  genes  = stressors.iloc[indexes_for_stressors, 0].to_numpy()
  y_train = mouse.iloc[:, -1].to_numpy().astype(int)
  y_train_for_correlation = mouse.iloc[:, 3]
  x_train[np.isnan(x_train)]=0

  return x_train,y_train,y_train_for_correlation, genes




def read_data():
    address  = "./" 
    dim = list()
    mouse1 = pd.read_csv(address + '240123_Pool1_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(mouse1.shape[0])
    mouse2 = pd.read_csv(address +'240123_Pool2_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[0] + mouse2.shape[0])
    mouse3 = pd.read_csv(address +'240123_Pool3_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[1] + mouse3.shape[0])
    mouse4 = pd.read_csv(address +'240123_Pool4_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[2] + mouse4.shape[0])
    mouse5 = pd.read_csv(address +'240123_Pool5_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[3] + mouse5.shape[0])
    mouse6 = pd.read_csv(address +'240123_Pool6_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[4] + mouse6.shape[0])
    mouse7 = pd.read_csv(address +'240123_Pool7_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[5] + mouse7.shape[0])
    mouse8 = pd.read_csv(address +'240123_Pool8_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[6] + mouse8.shape[0])
    mouse9 = pd.read_csv(address +'240123_Pool9_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[7] + mouse9.shape[0])
    mouse10 = pd.read_csv(address +'240123_Pool10_MouseZadj.csv', quoting=csv.QUOTE_NONE)
    dim.append(dim[8] + mouse10.shape[0])
    dim = np.asarray(dim)
    dim = dim.astype(int)

    stressors = pd.read_csv(address + '240303_DESeq_scaled_L2FCs_noMouse_unfiltered.csv')
    stressors = stressors.drop("ProductDescription",axis=1)
    stressors = stressors.drop("GeneName",axis=1)
    mouse = pd.concat([mouse1, mouse2, mouse3, mouse4, mouse5, mouse6, mouse7, mouse8, mouse9, mouse10])
    print(mouse.columns)
    print(mouse.shape[1])

    mouse['labels'] = mouse['Gene']
    return mouse,stressors,dim




def shuffle_mouse_for_labels(mouse,min,max):
    for i in range(0, 10):
        mouse_index = random.randint(1, 5)
        if i == 0:
            while (np.sum((mouse.iloc[0:dim[i], mouse_index]).isna()) > 300):
                mouse_index = random.randint(1, 5)
            mouse.iloc[np.where(mouse.iloc[0:dim[i], mouse_index] <= min)[0], -1] = 1
            mouse.iloc[np.where(mouse.iloc[0:dim[i], mouse_index] > max)[0], -1] = 2
            mouse.iloc[np.where((mouse.iloc[:,mouse_index]> min) & (mouse.iloc[:,mouse_index] <= max))[0],-1] = -1
        else:
            while (np.sum((mouse.iloc[dim[i - 1]:dim[i], mouse_index]).isna()) > 300):
                mouse_index = random.randint(1, 5)
            mouse.iloc[np.where(mouse.iloc[dim[i - 1]:dim[i]:, mouse_index].values <= min)[0] + dim[i - 1], -1] = 1
            mouse.iloc[np.where(mouse.iloc[dim[i - 1]:dim[i], mouse_index].values > max)[0] + dim[i - 1], -1] = 2
            mouse.iloc[np.where((mouse.iloc[:,mouse_index]> min) & (mouse.iloc[:,mouse_index] <= max))[0],-1] = -1
    return mouse



def shuffle_mouse_for_labels_test_on_boundary(mouse,min,max):
    for i in range(0, 10):
        mouse_index = random.randint(1, 5)
        if i == 0:
            while (np.sum((mouse.iloc[0:dim[i], mouse_index]).isna()) > 300):
                mouse_index = random.randint(1, 5)
            mouse.iloc[np.where(mouse.iloc[0:dim[i], mouse_index] <= min)[0], -1] = 0
            mouse.iloc[np.where(mouse.iloc[0:dim[i], mouse_index] > max)[0], -1] = 1
            mouse.iloc[np.where((mouse.iloc[:,mouse_index]> min) & (mouse.iloc[:,mouse_index] <= max))[0],-1] =-1
        else:
            while (np.sum((mouse.iloc[dim[i - 1]:dim[i], mouse_index]).isna()) > 300):
                mouse_index = random.randint(1, 5)
            mouse.iloc[np.where(mouse.iloc[dim[i - 1]:dim[i], mouse_index].values <= min)[0] + dim[i - 1], -1] = 0
            mouse.iloc[np.where(mouse.iloc[dim[i - 1]:dim[i], mouse_index].values > max)[0] + dim[i - 1], -1] = 1
            mouse.iloc[np.where((mouse.iloc[:,mouse_index]> min) & (mouse.iloc[:,mouse_index] <= max))[0],-1] =-1

    return mouse



def shuffle_mouse_for_labels_three_classes(mouse,range_a,range_b):
    for i in range(0, 10):
        mouse_index = random.randint(1, 5)

        if i == 0:
            while (np.sum((mouse.iloc[0:dim[i], mouse_index]).isna()) > 300):
                mouse_index = random.randint(1, 5)
            mouse.iloc[np.where(mouse.iloc[0:dim[i], mouse_index] >= range_b)[0], -1] = 2
            mouse.iloc[np.where((mouse.iloc[:,mouse_index]>= range_a) & (mouse.iloc[:,mouse_index] < range_b))[0],-1] = 1
            mouse.iloc[np.where(mouse.iloc[0:dim[i], mouse_index] < range_a)[0], -1] = 0

        else:
            while (np.sum((mouse.iloc[dim[i - 1]:dim[i], mouse_index]).isna()) > 100):
                mouse_index = random.randint(1, 5)
            mouse.iloc[np.where(mouse.iloc[dim[i - 1]:dim[i]:, mouse_index].values >= 0.5)[0] + dim[i - 1], -1] = 2
            mouse.iloc[np.where((mouse.iloc[:,mouse_index]>= -0.5) & (mouse.iloc[:,mouse_index] < 0.5))[0],-1] =1
            mouse.iloc[np.where(mouse.iloc[dim[i - 1]:dim[i], mouse_index].values < -0.5)[0] + dim[i - 1], -1] = 0
    
    return mouse

def train_and_leave_one_out_Test(x_train_origin,y_train_origin, genes=None, classifier=1):
    average_micro_accuracy = list()
    average_macro_accuracy = list()
    confusion_matrix_leave_one_out = list()
    dummy_conf_prior = list()
    dummy_conf_uniform = list()
    dummy_conf_stratified = list()
    
    importances_list = list()
    y_pred_cumulative = list()
    y_test_cumulative = list()

    microf1_prior   = list()
    macrof1_prior   = list()
    microf1_uniform = list()
    macrof1_uniform = list()
    microf1_stratified = list()
    macrof1_stratified = list()

    dummy_preds_prior_cumulative = list()
    dummy_preds_uniform_cumulative = list()
    dummy_preds_stratified_cumulative = list()

    misclassified_genes = list()
    test_len = 0

    for i in range(0, 10):
        if i == 0:
            x_test = x_train_origin[0:dim[i], :]
            y_test = y_train_origin[0:dim[i]]
            x_train = x_train_origin[dim[i]:, :]
            y_train = y_train_origin[dim[i]:]
            te_genes = genes[:dim[i]]
            tr_genes = genes[dim[i]:]
        else:
            x_test = x_train_origin[dim[i - 1]:dim[i], :]
            y_test = y_train_origin[dim[i - 1]:dim[i]]
            x_train = np.concatenate([x_train_origin[:dim[i - 1], :], x_train_origin[dim[i]:, :]])
            y_train = np.concatenate([y_train_origin[:dim[i - 1]], y_train_origin[dim[i]:]])
            te_genes = genes[dim[i - 1]:dim[i]]
            tr_genes = np.concatenate((genes[:dim[i - 1]], genes[dim[i]:]))

        x_train[np.isnan(x_train)] = 0


        # FORMER BEGIN-IF
        # if classifier ==1:
        num_negative = len(np.where(y_train==1)[0])
        num_positive = len(np.where(y_train==2)[0])
        regressor = RandomForestClassifier(n_estimators=100,
                                            random_state=1)
        
        dummy_prior     = DummyClassifier(strategy="prior")
        dummy_uniform   = DummyClassifier(strategy="uniform")
        dummy_stratified = DummyClassifier(strategy="stratified")

        thresholded =np.where(y_train!=-1)[0]
        
        x_train = x_train[thresholded]
        y_train = y_train[thresholded]
        thresholded =np.where(y_test!=-1)[0]
        x_test = x_test[thresholded]
        y_test = y_test[thresholded]
        tr_genes = tr_genes[thresholded]
        te_genes = te_genes[thresholded]
        stressors_import = list()

        regressor.fit(x_train, y_train)

        dummy_prior.fit(x_train, y_train)
        dummy_uniform.fit(x_train, y_train)
        dummy_stratified.fit(x_train, y_train)
        
        if not classifier:
            y_test_predicted2 = np.argmax(regressor.predict(x_test),axis=1)
        else:
            y_test_predicted2 = regressor.predict(x_test)
            y_pred_cumulative.append(y_test_predicted2)
            y_test_cumulative.append(y_test)

            dummy_preds_prior = dummy_prior.predict(x_test)
            dummy_preds_uniform = dummy_uniform.predict(x_test)
            dummy_preds_stratified = dummy_stratified.predict(x_test)
            dummy_preds_prior_cumulative.append(dummy_preds_prior)
            dummy_preds_uniform_cumulative.append(dummy_preds_uniform)
            dummy_preds_stratified_cumulative.append(dummy_preds_stratified)

            microf1_prior.append(f1_score(y_test, dummy_preds_prior, average="micro"))
            macrof1_prior.append(f1_score(y_test, dummy_preds_prior, average="macro"))

            microf1_uniform.append(f1_score(y_test, dummy_preds_uniform, average="micro"))
            macrof1_uniform.append(f1_score(y_test, dummy_preds_uniform, average="macro"))

            microf1_stratified.append(f1_score(y_test, dummy_preds_stratified, average="micro"))
            macrof1_stratified.append(f1_score(y_test, dummy_preds_stratified, average="macro"))



        misclassified_genes.append(te_genes[y_test_predicted2 != y_test])
        confusion_matrix_leave_one_out.append(confusion_matrix(y_test.astype(int), y_test_predicted2.astype(int)))
        dummy_conf_prior.append(confusion_matrix(y_test, dummy_preds_prior, normalize="pred"))
        dummy_conf_uniform.append(confusion_matrix(y_test, dummy_preds_uniform, normalize="pred"))
        dummy_conf_stratified.append(confusion_matrix(y_test, dummy_preds_stratified, normalize="pred"))
    

        cf_labels = ['Attenuated','Not Attenuated']
        plt.figure(figsize=(16, 16))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_leave_one_out[i])
        disp.plot(cmap='binary', values_format='.2f')
        disp.ax_.set_xticklabels(cf_labels)
        disp.ax_.set_yticklabels(cf_labels)
        disp.ax_.set_xlabel('True')
        disp.ax_.set_ylabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"IMAGES/ConfMat_Fold_{i}.svg", format="svg", dpi=1000)
        plt.close()


        if classifier ==0:
            average_micro_accuracy.append(f1_score(y_test_predicted2, y_test-1, average='micro'))
            average_macro_accuracy.append(f1_score(y_test_predicted2, y_test-1, average='macro'))
            importances = regressor.feature_importances_
            importances_list.append(importances)
        else:
            average_micro_accuracy.append(f1_score(y_test_predicted2, y_test, average='micro'))
            average_macro_accuracy.append(f1_score(y_test_predicted2, y_test, average='macro'))
            importances = regressor.feature_importances_
            importances_list.append(importances)


        test_len += len(y_test)


    print(test_len)
    confusion_matrix_leave_one_out = np.array(confusion_matrix_leave_one_out)
    dummy_conf_prior = np.array(dummy_conf_prior)
    dummy_conf_uniform = np.array(dummy_conf_uniform)
    dummy_conf_stratified = np.array(dummy_conf_stratified)
    misclassified_genes = np.concatenate(misclassified_genes)
    dummy_preds_prior_cumulative = np.concatenate(dummy_preds_prior_cumulative)
    dummy_preds_uniform_cumulative = np.concatenate(dummy_preds_uniform_cumulative)
    dummy_preds_stratified_cumulative = np.concatenate(dummy_preds_stratified_cumulative)
    y_pred_cumulative = np.concatenate(y_pred_cumulative)
    y_test_cumulative = np.concatenate(y_test_cumulative)
    
    microf1_prior = np.array(microf1_prior)
    macrof1_prior = np.array(macrof1_prior)
    microf1_uniform = np.array(microf1_uniform)
    macrof1_uniform = np.array(macrof1_uniform)
    microf1_stratified = np.array(microf1_stratified)
    macrof1_stratified = np.array(macrof1_stratified)

    avg_microf1_prior = microf1_prior.mean()
    avg_macrof1_prior = macrof1_prior.mean()
    avg_microf1_uniform = microf1_uniform.mean()
    avg_macrof1_uniform = macrof1_uniform.mean()
    avg_microf1_stratified = microf1_stratified.mean()
    avg_macrof1_stratified = macrof1_stratified.mean()

    print("PriorDummy (micro): ", avg_microf1_prior)
    print("PriorDummy (macro): ", avg_macrof1_prior)
    print("UniformDummy (micro): ", avg_microf1_uniform)
    print("UniformDummy (macro): ", avg_macrof1_uniform)
    print("StratifiedDummy (micro): ", avg_microf1_stratified)
    print("StratifiedDummy (macro): ", avg_macrof1_stratified)

    np.save("dummy_preds_prior_cumulative.npy", dummy_preds_prior_cumulative)
    np.save("dummy_preds_uniform_cumulative.npy", dummy_preds_uniform_cumulative)
    np.save("dummy_preds_stratified_cumulative.npy", dummy_preds_stratified_cumulative)
    np.save("y_test_cumulative.npy", y_test_cumulative)
    np.save("y_pred_cumulative.npy", y_pred_cumulative)
    np.save("misclassified_genes.npy", misclassified_genes)
    
    best_conf_mat_idx = confusion_matrix_leave_one_out.flatten()[::4].argmax()
    best_conf_mat = confusion_matrix_leave_one_out[best_conf_mat_idx]
    summed_conf_mat = confusion_matrix_leave_one_out.sum(axis=0)
    confusion_matrix_leave_one_out = confusion_matrix_leave_one_out.mean(axis=0)        
    dummy_conf_prior = dummy_conf_prior.mean(axis=0)
    dummy_conf_uniform = dummy_conf_uniform.mean(axis=0)
    dummy_conf_stratified = dummy_conf_stratified.mean(axis=0)

    cf_labels = ['Attenuated','Not Attenuated']
    plt.figure(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=dummy_conf_prior)
    disp.plot(cmap='binary', values_format='.2f')
    disp.ax_.set_xticklabels(cf_labels)
    disp.ax_.set_yticklabels(cf_labels)
    disp.ax_.set_xlabel('True')
    disp.ax_.set_ylabel('Predicted')
    plt.tight_layout()
    plt.savefig("IMAGES/DummyConfMat_ClassPrior.svg", format="svg", dpi=1000)
    plt.close()

    cf_labels = ['Attenuated','Not Attenuated']
    plt.figure(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=dummy_conf_uniform)
    disp.plot(cmap='binary', values_format='.2f')
    disp.ax_.set_xticklabels(cf_labels)
    disp.ax_.set_yticklabels(cf_labels)
    disp.ax_.set_xlabel('True')
    disp.ax_.set_ylabel('Predicted')
    plt.tight_layout()
    plt.savefig("IMAGES/DummyConfMat_UniformPrior.svg", format="svg", dpi=1000)
    plt.close()

    cf_labels = ['Attenuated','Not Attenuated']
    plt.figure(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=dummy_conf_stratified)
    disp.plot(cmap='binary', values_format='.2f')
    disp.ax_.set_xticklabels(cf_labels)
    disp.ax_.set_yticklabels(cf_labels)
    disp.ax_.set_xlabel('True')
    disp.ax_.set_ylabel('Predicted')
    plt.tight_layout()
    plt.savefig("IMAGES/DummyConfMat_StratifiedPrior.svg", format="svg", dpi=1000)
    plt.close()

    cf_labels = ['Attenuated','Not Attenuated']
    plt.figure(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=best_conf_mat)
    disp.plot(cmap='binary', values_format='.2f')
    disp.ax_.set_xticklabels(cf_labels)
    disp.ax_.set_yticklabels(cf_labels)
    disp.ax_.set_xlabel('True')
    disp.ax_.set_ylabel('Predicted')
    plt.tight_layout()
    plt.savefig("IMAGES/BestFoldUnnormalized.svg", format="svg", dpi=1000)
    plt.close()

    cf_labels = ['Attenuated','Not Attenuated']
    plt.figure(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=summed_conf_mat)
    disp.plot(cmap='binary', values_format='.2f')
    disp.ax_.set_xticklabels(cf_labels)
    disp.ax_.set_yticklabels(cf_labels)
    disp.ax_.set_xlabel('True')
    disp.ax_.set_ylabel('Predicted')
    plt.tight_layout()
    plt.savefig("IMAGES/CumulativeConfMat.svg", format="svg", dpi=1000)
    plt.close()

    return confusion_matrix_leave_one_out,average_micro_accuracy,average_macro_accuracy,importances_list


if __name__ == "__main__":

    mean_macro_acc = list()
    mean_micro_acc = list()
    feat = list()
    conf = list()

    mouse, stressors, dim = read_data()
    #
    mouse = shuffle_mouse_for_labels_test_on_boundary(mouse, -2, 0)

    x_train_origin, y_train_origin,y_train_for_correlation, genes = extract_train(mouse, stressors)

    num_features_original = x_train_origin.shape[1]

    #
    confusion_matrix_leave_one_out, average_micro_accuracy, average_macro_accuracy, importances_list = train_and_leave_one_out_Test(x_train_origin, y_train_origin, genes=genes, classifier=True)



    features,means,stds,orderings = return_importances_plot(importances_list, stressors, num_features_original)

    np.save("features.npy", features)
    np.save("means.npy", means)
    np.save("stds.npy", stds)
    np.save("orderings.npy", orderings)
    

    conf = get_confusion_matrix(confusion_matrix_leave_one_out, ['Attenuated','Not Attenuated'])
