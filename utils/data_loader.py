import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Graph connections for 17 body joints
edges = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [1, 7], [7, 8], [8, 9],
    [4, 12], [12, 13], [13, 14],
    [7, 10], [10, 11],
    [8, 15], [15, 16]
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


class PsyMoDataset:
    def __init__(self, data_dir="data", train_ids=None, test_ids=None):
        self.data_dir = data_dir
        self.skeletons_dir = os.path.join(data_dir, "semantic_data", "skeletons")
        self.metadata_file = os.path.join(data_dir, "metadata_raw_scores_v3.csv")

        if not os.path.isfile(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        # Read psychological scores from CSV
        self.scores_df = pd.read_csv(self.metadata_file)

        # Map RSE_Score to classes
        self.class_map = {"Low": 0, "Normal": 1, "High": 2}

        # Save IDs for training and testing
        if train_ids is None:
            self.train_ids = list(map(str, range(8)))  # IDs 0–7
        else:
            self.train_ids = list(map(str, train_ids))

        if test_ids is None:
            self.test_ids = list(map(str, range(8, 10)))  # IDs 8–9
        else:
            self.test_ids = list(map(str, test_ids))

    def rse_score_to_class(self, score):
        """Convert numerical RSE_Score to one of the three classes"""
        if score <= 10:
            return 'Low'
        elif score <= 20:
            return 'Normal'
        else:
            return 'High'

    def load_subject(self, subject_id):
        """Load all JSON files for a single participant"""
        subject_path = os.path.join(self.skeletons_dir, str(subject_id))
        if not os.path.isdir(subject_path):
            print(f"Subject {subject_id} not found.")
            return []

        skeleton_files = [f for f in os.listdir(subject_path) if f.endswith(".json")]
        dataset = []

        row = self.scores_df[self.scores_df["ID"] == int(subject_id)]
        if row.empty:
            label_str = "Normal"
        else:
            rse_score = row["RSE_Score"].values[0]
            label_str = self.rse_score_to_class(rse_score)

        y_label = self.class_map.get(label_str, 1)
        y_tensor = torch.tensor([y_label], dtype=torch.long)

        for file in skeleton_files:
            file_path = os.path.join(subject_path, file)
            with open(file_path, 'r') as f:
                frames = json.load(f)

            for frame in frames:
                keypoints_list = frame.get("keypoints", [])
                if len(keypoints_list) != 51:
                    continue  # Skip incomplete data

                # Convert to NumPy array
                keypoints = np.array(keypoints_list, dtype=np.float32).reshape(-1, 3)  # x, y, v
                valid_points = keypoints[keypoints[:, 2] > 0.5][:, :2]

                if len(valid_points) < 10:
                    continue  # Skip frames with too few points

                # Pad or truncate to exactly 17 points
                if len(valid_points) < 17:
                    mean_point = np.mean(valid_points, axis=0)
                    padded_points = np.vstack([valid_points] + [mean_point] * (17 - len(valid_points)))
                else:
                    padded_points = valid_points[:17]

                # Check for NaN values
                if np.isnan(padded_points).any():
                    print("NaN in keypoints — skipping frame")
                    continue

                x = torch.tensor(padded_points, dtype=torch.float32)
                data = Data(x=x, edge_index=edge_index, y=y_tensor, subject=int(subject_id))
                dataset.append(data)

        print(f"Subject {subject_id}: {len(dataset)} examples")
        return dataset

    def get_datasets(self):
        train_dataset = []
        test_dataset = []

        print("\n--- Loading Training Data ---")
        for sid in self.train_ids:
            train_dataset.extend(self.load_subject(sid))

        print("\n--- Loading Testing Data ---")
        for sid in self.test_ids:
            test_dataset.extend(self.load_subject(sid))

        return train_dataset, test_dataset
