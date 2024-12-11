import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_FILEPATH = "preprocess_data.json"

class ClothesPriceMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClothesPriceMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        self.fc4 = nn.Linear(int(hidden_size / 4), 1)

    def forward(self, x):
        x = F.dropout(self.fc1(x), p=0.1)
        x = F.relu(x)
        x = F.dropout(self.fc2(x), p=0.1)
        x = F.relu(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def predict(X):
    input_size = 31
    hidden_size = 1024
    model = ClothesPriceMLP(input_size, hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(r"UI\Model_Rerun\MLP\mlp_model.pth", map_location=torch.device(device)))

    model.eval()
    df = pd.read_json(DATA_FILEPATH)
    drop_columns = ['name', 'asin', 'url']
    df = df.drop(columns=drop_columns)

    # Extract the input and output
    X = df.drop(columns=["price"])  # Replace "price" with your target column name
    y = df["price"]
    

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_test = scaler.transform(X)
    # Convert DataFrame to a NumPy array and then to a PyTorch tensor
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0)  # Add a batch dimension

    # Get the model's prediction
    outputs = model(X_test)

    # Print the output as a float value
    return (float(outputs[0][0][0]))

print(predict(550))