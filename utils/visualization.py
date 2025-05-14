import matplotlib.pyplot as plt


def visualize_skeleton_sequence(skeleton_sequence, title="Skeleton Sequence"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for frame in skeleton_sequence:
        for j in range(len(frame)):
            ax.scatter(frame[j][0], frame[j][1], frame[j][2], c='b', s=10)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Run-level Evaluation")
    plt.show()
