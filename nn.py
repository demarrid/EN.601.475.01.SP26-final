from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

file_path = "KaggleV2-May-2016.csv"
# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "joniarroba/noshowappointments",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using device: {device}")

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 1, bias=True),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NN().to(device)
print(model)

## Convert to datetime
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

## Convert to Unix timestamp (seconds since 1970)
df["Scheduled_ts"] = df["ScheduledDay"].astype("int64") // 10**9
df["Appointment_ts"] = df["AppointmentDay"].astype("int64") // 10**9

## Add more timestamp feature
df["waiting_days"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
df["appointment_dayofweek"] = df["AppointmentDay"].dt.dayofweek
df["scheduled_hour"] = df["ScheduledDay"].dt.hour

## Add more age related features
df["is_child"] = (df["Age"] < 18).astype(int)
df["is_elderly"] = (df["Age"] >= 60).astype(int)

## Sanity Checks
df = df[df["waiting_days"] >= 0]
df = df[df["Age"] >= 0]

## Geneder encoding
df["Gender"] = df["Gender"].map({"F": 0, "M": 1})

## For Neighbourhood - Can use one-hot encoding or use frequency
freq = df["Neighbourhood"].value_counts(normalize=True)
df["Neighbourhood_freq"] = df["Neighbourhood"].map(freq)

## Medical feature comination
df["num_conditions"] = (df["Hipertension"] + df["Diabetes"] + df["Alcoholism"] + df["Handcap"])

## Target encoding
df["No-show"] = df["No-show"].map({"Yes": 1, "No": 0})

## Drop unused columns
df = df.drop(columns=[
    "PatientId",
    "AppointmentID",
    "ScheduledDay",
    "AppointmentDay",
    "Neighbourhood"
])

## Split to training and testing data
X = df.drop("No-show", axis=1)
y = df["No-show"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_test_t  = torch.tensor(X_test.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # (N,1)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)   # (N,1)
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)
train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False)
loss_fn = nn.BCEWithLogitsLoss()

print("Train label ratio:")
print(y_train.value_counts(normalize=True))

print("\nTest label ratio:")
print(y_test.value_counts(normalize=True))

# train model
learning_rate = 1e-3
batch_size = 64
epochs = 5

train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(X_test, batch_size=batch_size, shuffle=True)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    batch = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)              # shape (batch,1)
        loss = loss_fn(pred, y)

        # Backpropr
        loss.backward()
        optimizer.setop()
        optimizer.zero_grad()  # zero the gradient
        batch += 1
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    model.eval()
    #eval for batch normalziing and dropout
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()


    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n \tAccuracy:{(100*correct):>0.1f}%\n\tAverage loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Completed!")

with torch.no_grad():
    logits = model(X_test_t.to(device))
    y_pred = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
    f1 = f1_score(y_test_t.cpu().numpy(), y_pred)
    print(f"F1 score: {f1}")

print(f"Predicted class: {y_pred}")