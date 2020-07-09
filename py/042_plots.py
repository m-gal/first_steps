# -*- coding: utf-8 -*-
"""
Created at May 2020
    This code contains the function
    for plotting any results
@author: mikhail.galkin
"""
#%% #%% Importing required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

################################################################################
############################ G R A P H I C S ###################################
################################################################################
#%% Plot: loss function
def plot_loss(history, nn_name):
    plt.figure(figsize=(12, 6))
    # # Var#1: Use a log scale to show the wide range of values.
    # plt.semilogy(history.epoch,  history.history['loss'], label='Train '+nn_name)
    # plt.semilogy(history.epoch,  history.history['val_loss'], label='Val '+nn_name,
    #     linestyle="--")
    # Var#2: W\oo a log scaling.
    plt.plot(history.epoch, history.history["loss"], label="train " + nn_name)
    plt.plot(
        history.epoch,
        history.history["val_loss"],
        label="val " + nn_name,
        linestyle="--",
    )
    # Common parameters
    plt.grid(color="gray", linestyle=":", linewidth=0.3)
    plt.title(nn_name + ": model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


#%% Plot: resulting metrics
def plot_metrics(history, nn_name):
    plt.figure(figsize=(12, 6))
    metrics = ["accuracy", "auc", "precision", "recall"]
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(
            history.epoch, history.history["val_" + metric], linestyle="--", label="Val"
        )
        plt.xlabel("epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()
        # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


#%% Plot: confusion matrix
def plot_cm(target, prediction, data_name, nn_name, p=0.5):
    fig = plt.figure(figsize=(6, 5))
    cm = confusion_matrix(target, prediction > p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(nn_name + ": " + data_name + ": confusion matrix @{:.2f}".format(p))
    plt.ylabel("actual")
    plt.xlabel("predicted")
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


#%% Plot: ROC
def plot_roc(target, prediction, label_name):
    fp, tp, _ = roc_curve(target, prediction)
    plt.plot(100 * fp, 100 * tp, label=label_name, linewidth=2)
    plt.plot([0, 100], [0, 100], "r--")
    plt.xlabel("false positives [%]")
    plt.ylabel("true positives [%]")
    plt.xlim([-0.5, 100])
    plt.ylim([0, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.legend(loc="lower right")
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


#%% Plot: 3 ROC
def plot_3roc(
    train_target,
    train_prediction,
    val_target,
    val_prediction,
    test_target,
    test_prediction,
    nn_name,
):

    train_fp, train_tp, _ = roc_curve(train_target, train_prediction)
    train_auc = round(roc_auc_score(train_target, train_prediction), 5)
    val_fp, val_tp, _ = roc_curve(val_target, val_prediction)
    val_auc = round(roc_auc_score(val_target, val_prediction), 5)
    test_fp, test_tp, _ = roc_curve(test_target, test_prediction)
    test_auc = round(roc_auc_score(test_target, test_prediction), 5)

    plt.figure(figsize=(6, 6))
    plt.plot(
        100 * train_fp, 100 * train_tp, label="train:auc" + str(train_auc), linewidth=2
    )
    plt.plot(100 * val_fp, 100 * val_tp, label="val:auc" + str(val_auc), linewidth=2)
    plt.plot(
        100 * test_fp, 100 * test_tp, label="test:auc" + str(test_auc), linewidth=2
    )
    plt.plot([0, 100], [0, 100], "r--")
    plt.title(nn_name + ": ROC-AUC")
    plt.xlabel("false positives [%]")
    plt.ylabel("true positives [%]")
    plt.xlim([-0.5, 100])
    plt.ylim([0, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.legend(loc="lower right")
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


#%% Plot: Density
def plot_dens(train_target, train_prediction, test_target, test_prediction, nn_name):
    fig = plt.figure(figsize=(12, 6))
    d = pd.concat(
        [
            pd.DataFrame(
                {
                    "target": train_target.values,
                    "prediction": train_prediction.ravel(),
                    "data": "train",
                }
            ),
            pd.DataFrame(
                {
                    "target": test_target.values,
                    "prediction": test_prediction.ravel(),
                    "data": "test",
                }
            ),
        ]
    ).groupby(["target", "data"])

    q50_test_0 = d.quantile(q=0.50).values[0]
    q50_test_1 = d.quantile(q=0.50).values[2]
    q50_train_0 = d.quantile(q=0.50).values[1]
    q50_train_1 = d.quantile(q=0.50).values[3]

    d.prediction.plot.kde()
    plt.title(nn_name + ": density chart")
    plt.xlabel("predicted probability")
    plt.legend()

    plt.axvline(x=q50_test_0, ls="--", lw=1, c="tab:blue")
    plt.text(q50_test_0, 0, "Q50=" + str(q50_test_0.round(4)), rotation=90)
    plt.axvline(x=q50_train_0, ls="--", lw=1, c="tab:orange")

    plt.axvline(x=q50_test_1, ls="--", lw=1, c="tab:green")
    plt.text(q50_test_1 - 0.02, 0, "Q50=" + str(q50_test_1.round(4)), rotation=90)
    plt.axvline(x=q50_train_1, ls="--", lw=1, c="tab:red")

    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig
