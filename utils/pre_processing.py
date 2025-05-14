import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_skeleton_sequence(sequence):
    """
    Нормализует последовательность скелетных данных
    :param sequence: np.ndarray, shape = (T, J, C)
    :return: нормализованная последовательность
    """
    T, J, C = sequence.shape
    flat = sequence.reshape(-1, C)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(flat).reshape(T, J, C)
    return normalized


def create_graph_from_skeletons(skeleton_sequence, edge_index):
    """
    Преобразует последовательность скелетов в список Graph Data объектов
    """
    graph_data_list = []
    for frame in skeleton_sequence:
        x = torch.tensor(frame, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        graph_data_list.append(data)
    return graph_data_list
