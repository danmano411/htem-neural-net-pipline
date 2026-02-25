import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn

import matplotlib.pyplot as plt

cwd = Path.cwd()
PARQUET_PATH = cwd / "datasets" / "htem_neural_net_dataset.parquet"
# print(PARQUET_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

# ============================ Hyperparameters ========================================
train_size = 0.8
batch_size = 128 
num_epochs = 2000
learning_rate = 0.001
dropoutP = 0.2 # fraction of values in last layer to be zeroed

hidden_layer_sizes = [2, 4, 8, 16, 16, 8]
output_size = 1

# ========================== Load and preprocess data ================================

print("Loading data ...")

class HTEMDataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None):
        df = pd.read_parquet(dataset_path)

        self.X = torch.from_numpy(df.iloc[:, 3:].to_numpy()).float()
        self.y = torch.from_numpy(df.iloc[:, 2].to_numpy()).unsqueeze(1).float()

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

htem_dataset = HTEMDataset(PARQUET_PATH)

# Split datasets
N = len(htem_dataset)
n_train = int(train_size * N)
n_test = N - n_train

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split (
    dataset=htem_dataset, 
    lengths=[n_train, n_test],
    generator=generator
)

# Normalize dataset (X)
mean = htem_dataset.X[train_dataset.indices].mean(dim=0)
std = htem_dataset.X[train_dataset.indices].std(dim=0, unbiased=False).clamp_min(1e-8)
htem_dataset.X = (htem_dataset.X - mean)/std

# Construct Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========================== Construct Model ======================================

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropP):
        super().__init__()

        layers = []
        prev_size = input_size

        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        # Add dropout on last layer (before output)
        layers.append(nn.Dropout(p=dropP))

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)   
    
    def forward(self, x):
        return self.model(x)
    
print("Creating model ...")

model = NeuralNetwork(htem_dataset.X.shape[1], hidden_layer_sizes, output_size, dropoutP).to(device)
# print(model)
total_parameters = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {total_parameters}")
# total_parameters_expression = " + ".join(str(p.numel()) for p in model.parameters())
# print(total_parameters_expression)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ======================== Evaluation ========================

def evaluate_model(dataloader):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute MSE
        MSE = ((all_outputs - all_labels) ** 2).mean().item()
        
        # Compute R^2
        y_mean = all_labels.mean()
        SS_res = ((all_labels - all_outputs) ** 2).sum()
        SS_tot = ((all_labels - y_mean) ** 2).sum()
        
        if SS_tot.item() == 0:
            r2_score = float("nan")
        else:
            r2_score = (1 - SS_res / SS_tot).item()

        if was_training:
            model.train()
        return MSE, r2_score

train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # Unshuffled

# ======================== Live Plot ========================

plt.ion()  # interactive mode on
fig, ax = plt.subplots()

train_mse_history = []
test_mse_history = []
epochs_history = []

(train_line,) = ax.plot([], [], label="Train MSE")
(test_line,) = ax.plot([], [], label="Test MSE")

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Train vs Test MSE (Live)")
ax.legend()
ax.grid(True)

fig.canvas.manager.set_window_title("HTEM Training Monitor")
fig.show()
fig.canvas.draw()

# ======================== Training ========================
print("Training model ...")

model.train()
for epoch in range(num_epochs):

    for data, labels in train_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        ### Plot Logic (model.eval() called in evaluate_model and model.train() restored)
        train_MSE, _ = evaluate_model(train_eval_loader)
        test_MSE, _ = evaluate_model(test_dataloader)

        ep = epoch + 1
        epochs_history.append(ep)
        train_mse_history.append(train_MSE)
        test_mse_history.append(test_MSE)

        train_line.set_data(epochs_history, train_mse_history)
        test_line.set_data(epochs_history, test_mse_history)

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw_idle()
        plt.pause(0.001)  

        ### Plot logic end
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train MSE: {train_MSE:.6f}, Test MSE: {test_MSE:.6f}")

# ======================== Conclusion ========================

model.eval()
train_MSE, train_r2_score = evaluate_model(train_eval_loader)
test_MSE, test_r2_score = evaluate_model(test_dataloader)

print(f'Train MSE: {train_MSE:.6f}, Train R^2: {train_r2_score:.4f}')
print(f'Test MSE: {test_MSE:.6f}, Test R^2: {test_r2_score:.4f}')

# ======================== Save Result =========================
# save batch_size, epochs, hidden_layer_sizes, total_parameters, train_MSE, train_r2_score, test_MSE, test_r2_score
save_flag = input("Save to Tests? (y/n): ").strip().lower()
while save_flag not in ("y", "n"):
    save_flag = input("Save to Tests? (y/n): ").strip().lower()

if save_flag == "y":
    # Path
    TESTS_CSV_PATH = cwd / "tests" / "htem_neural_net_tests.csv"
    TESTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Round metrics
    train_MSE = round(train_MSE, 6)
    test_MSE = round(test_MSE, 6)
    train_r2_score = round(train_r2_score, 4)
    test_r2_score = round(test_r2_score, 4)

    # Row data
    row = {
        "batch_size": batch_size,
        "epochs": num_epochs,
        "hidden_layer_sizes": str(hidden_layer_sizes),
        "total_parameters": total_parameters,
        "train_MSE": train_MSE,
        "train_r2_score": train_r2_score,
        "test_MSE": test_MSE,
        "test_r2_score": test_r2_score,
    }

    # Load or create CSV
    if TESTS_CSV_PATH.exists():
        df = pd.read_csv(TESTS_CSV_PATH)
        trial_num = len(df) + 1
    else:
        df = pd.DataFrame(columns=["trial_num", *row.keys()])
        trial_num = 1

    # Append
    row["trial_num"] = trial_num
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TESTS_CSV_PATH, index=False)

    print(f"Saved trial {trial_num} to tests")
else:
    print("Save cancelled")