import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import math, shutil, time
from datetime import datetime

cwd = Path.cwd()
PARQUET_PATH = cwd / "datasets" / "htem_neural_net_dataset.parquet"
# print(PARQUET_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

# Path
TESTS_CSV_PATH = cwd / "tests" / "architecture_comparisons_tests.csv"
TESTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================ Hyperparameters ========================================
# Constants throughout all tests
train_size = 0.8
batch_size = 256 
learning_rate = 0.001
dropoutP = 0.2 # fraction of values in last layer to be zeroed
output_size = 1

hidden_layer_size_grid = [
    # [2, 4, 16, 32, 16], # 3955
    [2, 4, 8, 32, 32, 16], # 4715
    [4, 8, 16, 8], # 5877
    # [4, 8, 16, 32, 16], # 6821
    # [4, 8, 16, 32, 32, 8] # 7605
]

num_epochs = 10000

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

# Get train test split
N = len(htem_dataset)
n_train = int(train_size * N)
n_test = N - n_train

# Split datasets
split_seed = 42
g_split = torch.Generator().manual_seed(split_seed)
train_dataset, test_dataset = random_split(htem_dataset, [n_train, n_test], generator=g_split)

# Normalize dataset (X)
mean = htem_dataset.X[train_dataset.indices].mean(dim=0)
std = htem_dataset.X[train_dataset.indices].std(dim=0, unbiased=False).clamp_min(1e-8)
htem_dataset.X = (htem_dataset.X - mean)/std

# ========================== Model Setup ======================================

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropP, batchNorm):
        super().__init__()

        layers = []
        prev_size = input_size

        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            if batchNorm: layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            prev_size = size

        # Add dropout on last layer (before output)
        if dropP != 0.0:
            layers.append(nn.Dropout(p=dropP))

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)   
    
    def forward(self, x):
        return self.model(x)

# ======================== Evaluation ================================

def evaluate_model(model, dataloader):
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

        # Compute RMSE
        RMSE = math.sqrt(MSE)

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

        return MSE, r2_score, RMSE, all_labels, all_outputs

# ======================== Grid Hyperparameter Search ========================

start = time.perf_counter()

for i, hidden_layer_sizes in enumerate(hidden_layer_size_grid):
    for n in range(4):
        # ================================ Model Creation ===================================
        print("Creating model ...")

        # Trial seed
        trial_seed = 715

        # Construct Dataloaders
        g_loader = torch.Generator().manual_seed(trial_seed)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g_loader)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=g_loader)
        train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=g_loader)

        effective_dropout = dropoutP if n in [1, 3] else 0.0
        has_batch_norm = True if n in [2, 3] else False

        model = NeuralNetwork(
            input_size=htem_dataset.X.shape[1],
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=output_size,
            dropP=effective_dropout,
            batchNorm=has_batch_norm
        ).to(device)

        print(model)
        
        total_parameters = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_parameters}")

        # total_parameters_expression = " + ".join(str(p.numel()) for p in model.parameters())
        # print(total_parameters_expression)

        # Epoch Metric storage
        epoch_array = []
        train_MSE_array = []
        test_MSE_array = []
        train_RMSE_array = []
        test_RMSE_array = []
        train_r2_score_array = []
        test_r2_score_array = []

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
                ### model.eval() called in evaluate_model and model.train() restored
                train_MSE, train_r2_score, train_RMSE, _, _ = evaluate_model(model, train_eval_loader)
                test_MSE, test_r2_score, test_RMSE, _, _ = evaluate_model(model, test_dataloader)

                epoch_array.append(epoch+1)

                train_MSE_array.append(train_MSE)
                train_RMSE_array.append(train_RMSE)
                train_r2_score_array.append(train_r2_score)

                test_MSE_array.append(test_MSE)
                test_RMSE_array.append(test_RMSE)
                test_r2_score_array.append(test_r2_score)
                
                print(f"Epoch [{epoch+1}/{num_epochs}], Train MSE: {train_MSE:.6f}, Test MSE: {test_MSE:.6f}, Train RMSE: {train_RMSE:.6f}, Test RMSE: {test_RMSE:.6f}")

        # ======================== Conclusion ========================

        model.eval()
        _, _, _, train_datay, train_modely = evaluate_model(model, train_eval_loader)
        _, _, _, test_datay, test_modely = evaluate_model(model, test_dataloader)

        # ======================== Save Result =========================

        # Covert (N, 1) to (N,)
        train_datay  = [x[0] for x in train_datay.tolist()]
        train_modely = [x[0] for x in train_modely.tolist()]
        test_datay   = [x[0] for x in test_datay.tolist()]
        test_modely  = [x[0] for x in test_modely.tolist()]

        # Row data
        row = {
            "total_epochs": num_epochs,
            "split_seed": split_seed,
            "trial_seed": trial_seed,
            "total_parameters": total_parameters,
            "hidden_layer_sizes": hidden_layer_sizes,
            "dropout_size": effective_dropout,
            "has_batch_normalization": has_batch_norm,
            "epoch_list": epoch_array,
            "train_MSE": train_MSE_array,
            "test_MSE": test_MSE_array,
            "train_RMSE": train_RMSE_array,
            "test_RMSE": test_RMSE_array,
            "train_r2_score": train_r2_score_array,
            "test_r2_score": test_r2_score_array,
            "train_y": train_datay,
            "train_model_y": train_modely,
            "test_y": test_datay,
            "test_model_y": test_modely,
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

        # ================= Backup existing CSV =================
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_dir = TESTS_CSV_PATH.parent / "backups" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_path = backup_dir / TESTS_CSV_PATH.name
        shutil.copy2(TESTS_CSV_PATH, backup_path)

        print(f"Backup saved to {backup_path}")

# ================= Print Final Time Logic =================
elapsed = time.perf_counter() - start

hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")