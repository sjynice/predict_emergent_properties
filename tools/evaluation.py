import numpy as np
from scipy.stats import spearmanr
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, accuracy_score, f1_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score,
    log_loss, top_k_accuracy_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix
)

 
def eval_regression(y_true, y_pred, eps=1e-8):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.std(y_true) + eps)

    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "NRMSE": nrmse,
        "SpearmanR": spearmanr(y_true, y_pred).correlation,
        "PearsonR": np.corrcoef(y_true, y_pred)[0, 1],
    }

def eval_binary(y_true, y_prob, threshold=None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Default is 0.5; can also be grid-searched externally
    if threshold is None:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)

    p_pos = np.mean(y_true)
    auprc = average_precision_score(y_true, y_prob)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),

        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),  # Very robust

        "AUC": roc_auc_score(y_true, y_prob),
        "AUPRC": auprc,

        "AUPRC_rel": (auprc - p_pos) / p_pos if p_pos > 0 else np.nan,

        "Brier": brier_score_loss(y_true, y_prob),
        "pos_rate": p_pos
    }

def eval_multiclass(n_class, y_true, y_prob, top_k=(2,), ordinal_metrics=True):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    class_dist = np.bincount(y_true, minlength=n_class) / len(y_true)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),

        "F1_macro": f1_score(y_true, y_pred, average='macro'),
        "F1_weighted": f1_score(y_true, y_pred, average='weighted'),

        "class_dist": class_dist.tolist(),
    }

    # Top-k accuracy
    for k in top_k:
        metrics[f"Top{k}_Acc"] = top_k_accuracy_score(
            y_true, y_prob, k=k, labels=np.arange(n_class)
        )
        
    if ordinal_metrics:
        # True number of steady states per class: 1, 2, ..., n_class
        classes = np.arange(1, n_class + 1)

        # softmax expected value
        y_pred_expect = np.sum(y_prob * classes, axis=1)

        # Convert back to true steady-state count
        y_true_ord = y_true + 1
        y_pred_ord = y_pred + 1

        metrics.update({
            # Based on probability expectation (most recommended)
            "MAE_ord_expect": mean_absolute_error(y_true_ord, y_pred_expect),
            "RMSE_ord_expect": np.sqrt(
                mean_squared_error(y_true_ord, y_pred_expect)
            ),
            "R2_ord_expect": r2_score(y_true_ord, y_pred_expect),

            # Based on argmax (intuitive but coarse)
            "MAE_ord_argmax": mean_absolute_error(y_true_ord, y_pred_ord),
            "RMSE_ord_argmax": np.sqrt(
                mean_squared_error(y_true_ord, y_pred_ord)
            ),
        })

    return metrics

def ScatterPlot(y_te, y_pred, pdfname):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7
    })

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        ["#F8D8CC", "#E95920"]
    )

    y_te = np.asarray(y_te)
    y_pred = np.asarray(y_pred)

    # ===== If vector output, take only the 0th dimension =====
    if y_te.ndim > 1:
        figy_true = y_te[:, 0]
    else:
        figy_true = y_te

    if y_pred.ndim > 1:
        figy_pred = y_pred[:, 0]
    else:
        figy_pred = y_pred

    # ===== Remove nan/inf =====
    mask = (
        np.isfinite(figy_true) &
        np.isfinite(figy_pred)
    )

    figy_true = figy_true[mask]
    figy_pred = figy_pred[mask]

    xy = np.vstack([figy_true, figy_pred])
    z = gaussian_kde(xy)(xy)  # Density value at each point

    # ==== Sort by density (high-density points drawn last to avoid occlusion) ====
    idx = np.argsort(z)
    x, y, z = figy_true[idx], figy_pred[idx], z[idx]

    # ==== Draw density scatter plot ====
    plt.figure(figsize=(3, 3), dpi=100)
    # plt.scatter(x, y, c=z, s=10, cmap='Reds', edgecolors='none', alpha=0.9)

    z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))
    threshold = 0.2  # Threshold (between 0 and 1, adjustable)

    # Clip
    z_clipped = np.clip(z_norm, 0, threshold) / threshold
    #z_stretch = z_norm ** 0.1  # Smaller value gives stronger contrast
    plt.scatter(x, y, c=z_clipped, vmin=0, vmax=1, s=10, cmap=cmap, edgecolors='none', alpha=1)

    # ==== Draw ideal prediction line ====
    plt.plot([0, 1], [0, 1], 'r--', lw=1)

    # ==== Axis ranges and labels ====
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.xlim(1, 3)
    # plt.ylim(1, 3)
    # plt.xlabel("True abundance of species 1", fontsize=9)
    # plt.ylabel("Predicted abundance of species 1", fontsize=9)
    plt.tick_params(direction='in')

    plt.savefig(
        pdfname,
        dpi=400,
        bbox_inches="tight",
        transparent=True
    )
    plt.tight_layout()
    plt.show()

def ConfusionMatrix(species_idx, y_te, y_pr, pdfname): # y_pr is a matrix containing predictions for all species; y_pred later is the single-species prediction result
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7
    })

    y_te = np.asarray(y_te)
    y_pr = np.asarray(y_pr)

    # Auto-compatible:
    # (N,10) -> take species_idx
    # (N,) / (N,1) -> use directly

    if y_te.ndim == 2 and y_te.shape[1] > 1:
        y_true = y_te[:, species_idx]
    else:
        y_true = y_te.ravel()

    if y_pr.ndim == 2 and y_pr.shape[1] > 1:
        y_pred = y_pr[:, species_idx]
    else:
        y_pred = y_pr.ravel()

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    colors = ["#F7845600", "#F92702DB"]

    my_cmap = LinearSegmentedColormap.from_list('my_cmap',colors)

    plt.figure(figsize=(6, 5))

    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=my_cmap,
        vmin=0,
        vmax=1
    )

    # class_labels = [
    #     'Non-core species',
    #     'Core species'
    # ]

    class_labels = [
        'Unfluctuated',
        'Fluctuated'
    ]

    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.tick_params(axis='both',which='both',length=0)

    data = ax.collections[0].get_array().reshape(cm_norm.shape)
    threshold = 0.5
    for idx, text in enumerate(ax.texts):

        row = idx // cm_norm.shape[1]
        col = idx % cm_norm.shape[1]
        value = data[row, col]
        text.set_color(
            'white' if value > threshold else 'black'
        )

    for spine in ax.spines.values():

        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.title(
    #     f'Species {species_idx}\nP(Predicted | True)'
    # )
    plt.savefig(
        pdfname,
        dpi=400,
        bbox_inches="tight",
        transparent=True
    )
    plt.show()