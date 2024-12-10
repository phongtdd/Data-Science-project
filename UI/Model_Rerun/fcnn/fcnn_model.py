import torch
import torch.nn as nn
import pandas as pd

def fcnn_predict(index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class FCNN(nn.Module):
        def __init__(self, input_dim):
            super(FCNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(0.01),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.01),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.01),
                nn.BatchNorm1d(128),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.01),
                nn.BatchNorm1d(64),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.model(x)

    df = pd.read_pickle(r'UI\Model_Rerun\fcnn\fcnn_processed_data.pkl')

    input_dim = df.shape[1] - 1
    model = FCNN(input_dim).to(device)
    model.load_state_dict(torch.load(r'UI\Model_Rerun\fcnn\fcnn_model.pth'))
    model.eval()

    X_index = df.drop("price", axis=1).iloc[index].to_numpy()
    X_tensor_index = torch.tensor(X_index, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred_index = model(X_tensor_index).cpu().numpy().flatten()
    return y_pred_index[0]

print(fcnn_predict(0))