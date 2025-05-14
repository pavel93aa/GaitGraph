import torch
from sklearn.preprocessing import StandardScaler


def normalize_skeleton_sequence(sequence):
    """
    Normalizes a sequence of skeleton data.

    :param sequence: np.ndarray, shape = (T, J, C)
    :return: normalized skeleton sequence
    """
    T, J, C = sequence.shape
    flat = sequence.reshape(-1, C)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(flat).reshape(T, J, C)
    return normalized


def create_graph_from_skeletons(skeleton_sequence, edge_index):
    """
    Converts a skeleton sequence into a list of Graph Data objects.

    :param skeleton_sequence: list or array of skeleton frames
    :param edge_index: tensor of graph edges
    :return: list of PyG Data objects
    """
    graph_data_list = []
    for frame in skeleton_sequence:
        x = torch.tensor(frame, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        graph_data_list.append(data)
    return graph_data_list
