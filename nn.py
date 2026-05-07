from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from main import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import utils

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
THRESHOLD = 0.44

print(f"Using device: {device}")

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(15, 12, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(12, 12, bias=True),
            nn.ReLU(),  
            nn.Linear(12, 1, bias=True),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    batch_size = dataloader.batch_size
    size = len(dataloader.dataset)
    model.train()
    batch = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)              # shape (batch,1)
        loss = loss_fn(pred, y)

        # Backpropr
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # zero the gradient
        batch += 1
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    #eval for batch normalziing and dropout
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            probs = torch.sigmoid(pred)
            pred_labels = (probs > THRESHOLD).float()
            correct += (pred_labels == y).float().sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n \tAccuracy:{(100*correct):>0.1f}%\n\tAverage loss: {test_loss:>8f} \n")

def train_nn():
    learning_rate = 1e-3
    epochs = 24
    batch_size = 128
    pos_weight = 2.5

    model = NN().to(device)
    print(model)
    X_train, X_test, y_train, y_test = get_data()
    X_train_model = X_train.drop(columns=["Scheduled_ts", "Appointment_ts"], errors="ignore").copy()
    X_test_model = X_test.drop(columns=["Scheduled_ts", "Appointment_ts"], errors="ignore").copy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_model)
    X_test_scaled = scaler.transform(X_test_model)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # (N,1)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)   # (N,1)
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    return model, scaler

def evaluate_fairness(model, scaler):
    X_train, X_test, y_train, y_test = get_data()
    X_test_model = X_test.drop(columns=["Scheduled_ts", "Appointment_ts"], errors="ignore").copy()
    model.eval()
    X_test_scaled = scaler.transform(X_test_model)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs > THRESHOLD).astype(int)
    
    print(probs.min(), probs.max())
    print(preds.mean())
    overall_precision = precision_score(y_test.values, preds, zero_division=0)
    overall_recall = recall_score(y_test.values, preds, zero_division=0)
    overall_f1 = f1_score(y_test.values, preds, zero_division=0)
    print(
        "Overall metrics:\n"
        f"\tPrecision: {overall_precision:.4f}\n"
        f"\tRecall: {overall_recall:.4f}\n"
        f"\tF1: {overall_f1:.4f}"
    )
    utils.evaluate_gender_fairness(X_test, y_test, preds, probs)
    utils.evaluate_age_fairness(X_test, y_test, preds, probs)


if __name__ == "__main__":
    model, scaler = train_nn()
    print("Completed!")

    evaluate_fairness(model, scaler)