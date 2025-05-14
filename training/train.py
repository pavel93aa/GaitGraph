import sys
import os
import torch
from torch_geometric.data import DataLoader, Batch
from models.gaitgraph import GaitGraph
from utils.data_loader import PsyMoDataset

# Add root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

if __name__ == "__main__":
    loader = PsyMoDataset(data_dir="data")
    train_data, test_data = loader.get_datasets()

    if len(train_data) == 0:
        print("No data for training!")
    if len(test_data) == 0:
        print("No data for evaluation!")

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = GaitGraph()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    loss_history = []

    print("\n--- Training Started ---")
    patience = 5
    best_loss = float('inf')
    counter = 0

    for epoch in range(1, 21):  # 20 epochs
        model.train()
        total_loss = 0
        count = 0

        for data in train_loader:
            if data.x is None or data.x.shape[0] == 0:
                continue  # Skip empty graphs

            # Create batch
            batched_data = Batch.from_data_list([data])

            optimizer.zero_grad()
            out = model(batched_data)

            # Check for NaN/inf in output
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("Output contains NaN or inf — skipping step")
                continue

            loss = criterion(out, batched_data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else float('inf')
        loss_history.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping at epoch", epoch)
                break
