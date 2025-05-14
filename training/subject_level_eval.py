import torch
from collections import Counter
from sklearn.metrics import classification_report


def subject_level_evaluate(loader, model):
    model.eval()
    subject_predictions = {}

    with torch.no_grad():
        for data in loader:
            sid = data.subject.item()
            pred = model(data).argmax(dim=1).item()

            if sid not in subject_predictions:
                subject_predictions[sid] = []
            subject_predictions[sid].append(pred)

    final_pred = {}
    final_true = []

    for sid, preds in subject_predictions.items():
        majority_class = Counter(preds).most_common(1)[0][0]
        final_pred[sid] = majority_class
        final_true.append(data.y.item())

    print("\n--- Subject-Level Evaluation ---")
    print(classification_report(final_true, list(final_pred.values()), digits=4))
