import torch
from torch_geometric.data import DataLoader
from models.gaitgraph import GaitGraph
from utils.data_loader import PsyMoDataset

# Загрузка данных
loader = PsyMoDataset(data_dir="../data")
train_data, test_data = loader.get_datasets()

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Модель
model = GaitGraph()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


# Цикл обучения
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


# Обучение
for epoch in range(1, 21):  # 20 эпох
    loss = train()
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
