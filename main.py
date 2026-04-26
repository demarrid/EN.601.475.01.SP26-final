
import kagglehub
from kagglehub import KaggleDatasetAdapter
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

def get_data():
    return X_train, X_test, y_train, y_test