import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class ClothesPriceMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClothesPriceMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.fc4 = nn.Linear(int(hidden_size/4), int(hidden_size/8))  
        self.fc5 = nn.Linear(int(hidden_size/8), 1)  
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(int(hidden_size / 2))
        self.bn3 = nn.BatchNorm1d(int(hidden_size / 4))
        self.bn4 = nn.BatchNorm1d(int(hidden_size / 8))

    def forward(self, x):
        x = F.dropout(self.fc1(x),p=0.2)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.dropout(self.fc2(x),p=0.2)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.dropout(self.fc3(x),p=0.2)
        x = F.relu(x)
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.fc5(x)
        return x

def predict(X):
    input_size = 31
    hidden_size = 2048
    model = ClothesPriceMLP(input_size, hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(r"UI\Model_Rerun\MLP\mlp_model.pth", map_location=torch.device(device)))

    model.eval()
    
    X_test = X
    scaler = joblib.load('UI\Model_Rerun\MLP\scaler.pkl')
    X_test = scaler.transform(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test = X_test.squeeze()  # Removes the extra dimension if it's 1
    X_test = X_test.unsqueeze(0)  # Add the batch dimension explicitly
    outputs = model(X_test)

    # Print the output as a float value
    return (float(outputs))