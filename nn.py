from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from main import get_data

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using device: {device}")

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True),
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
            probs = torch.sigmoid(pred)
            pred_labels = (probs > 0.5).float()
            correct += (pred_labels == y).float().sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n \tAccuracy:{(100*correct):>0.1f}%\n\tAverage loss: {test_loss:>8f} \n")

print("Completed!")

def train_nn():
    # train model parameters
    learning_rate = 1e-3
    epochs = 5
    batch_size = 64

    model = NN().to(device)
    print(model)
    X_train, X_test, y_train, y_test = get_data()

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # (N,1)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)   # (N,1)
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    return model


def evaluate_fairness(model):
    X_train, X_test, y_train, y_test = get_data()

if __name__ == "__main__":
    model = train_nn()
    print("Completed!")

    evaluate_fairness(model)