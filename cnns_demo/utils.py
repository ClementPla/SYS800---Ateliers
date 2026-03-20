import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_confusion_matrix(confMat, show=True, class_names=None):
    num_classes = confMat.shape[0]
    plt.figure(figsize=(8, 8))
    plt.imshow(confMat, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{confMat[i, j]}", ha="center", va="center", color="white" if confMat[i, j] > 0.5 else "black")
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar(fraction=0.046, pad=0.04).set_label('Ratio', rotation=270, labelpad=20)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names if class_names is not None else tick_marks, rotation=45)
    plt.yticks(tick_marks, class_names if class_names is not None else tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    accuracy = np.trace(confMat) / np.sum(confMat) if np.sum(confMat) > 0 else 0
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.2%}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    if show:
        plt.show()

def compute_and_plot_confusion_matrix(pred, gt, num_classes=26, show=True, class_names=None):
    
    # Conversion propre en numpy
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy().flatten()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy().flatten()
    
    # Construction de la matrice
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for p, g in zip(pred, gt):
        if 0 <= p < num_classes and 0 <= g < num_classes:
            conf_matrix[g, p] += 1
    
    # --- CORRECTION : Gestion de la division par zéro ---
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    # On ne divise que là où la somme est > 0, sinon on laisse 0
    conf_matrix = np.divide(conf_matrix, row_sums, out=np.zeros_like(conf_matrix), where=row_sums != 0)

    plot_confusion_matrix(conf_matrix, show=show, class_names=class_names)