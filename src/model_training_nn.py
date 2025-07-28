import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from src.utils import Utility
from sklearn.metrics import mean_squared_error
from model_training_xgboost import R2_Adjusted

logger = Utility().setup_logger()

class BatteryDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BatteryNeuralNetwork(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout1d(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout1d(0.1),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    params = Utility().read_params()
    random_state = params['General']['random_state']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    df = pd.read_csv(os.path.join(root_dir, 'Data', 'processed', 'processed_trip_data.csv'))

    X = df.drop('SoC', axis=1).values
    y = df['SoC'].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)

    train_dataset = BatteryDataset(X_train, y_train)
    test_dataset = BatteryDataset(X_test, y_test)
    val_dataset = BatteryDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_dim = X.shape[1]
    model = BatteryNeuralNetwork(input_dim)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(1)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch+1}, Loss: {loss.item(): .4f}")
        print(f"Epoch {epoch+1}, Loss: {loss.item(): .4f}")

    model.eval()

    r2_obj = R2_Adjusted()

    predictions_train = []
    targets_train = []

    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X).squeeze()
            predictions_train.extend(outputs.tolist())
            targets_train.extend(batch_y.tolist())

    mse_train = mean_squared_error(targets_train, predictions_train)
    r2_adj_train = r2_obj.adjusted_r2_score(targets_train, predictions_train, df.shape[1]-1)

    logger.info(f"Train MSE: {mse_train: .4f}, Train r2 adjusted score: {r2_adj_train: .4f}")

    predictions_test = []
    targets_test = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            predictions_test.extend(outputs.tolist())
            targets_test.extend(batch_y.tolist())

    mse_test = mean_squared_error(targets_test, predictions_test)
    r2_adj_test = r2_obj.adjusted_r2_score(targets_test, predictions_test, df.shape[1]-1)

    logger.info(f"Test MSE: {mse_test: .4f}, Test r2 adjusted score: {r2_adj_test: .4f}")

    predictions_val = []
    targets_val = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X).squeeze()
            predictions_val.extend(outputs.tolist())
            targets_val.extend(batch_y.tolist())

    mse_val = mean_squared_error(targets_val, predictions_val)
    r2_adj_val = r2_obj.adjusted_r2_score(targets_val, predictions_val, df.shape[1]-1)

    logger.info(f"Validation MSE: {mse_val: .4f}, Validation r2 adjusted score: {r2_adj_val: .4f}")

    os.makedirs(os.path.join(root_dir, 'Models'),exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'Metrics'), exist_ok=True)

    metrics = {
        'mse_train':mse_train,
        'r2_adj_train':r2_adj_train,
        'mse_test': mse_test,
        'r2_adj_test': r2_adj_test,
        'mse_val': mse_val,
        'r2_adj_val': r2_adj_val
    }

    with open(os.path.join(root_dir, 'Metrics', 'nn_metrics.json'), 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    torch.save(model.state_dict(), os.path.join(root_dir, 'Models', 'nn_model.pth'))

