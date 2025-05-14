from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch


def evaluate(loader, model):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data in loader:
            if data.x.shape[0] != 17:
                continue  # Skip invalid examples

            batched_data = Batch.from_data_list([data])
            out = model(batched_data)
            pred = out.argmax(dim=1)

            all_preds.extend(pred.tolist())
            all_true.extend(batched_data.y.tolist())

    print("\n--- Run-Level Evaluation ---")
    print(classification_report(all_true, all_preds, digits=4))

    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
    plt.title("Confusion Matrix - Run Level")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
