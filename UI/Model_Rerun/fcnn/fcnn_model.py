import torch
import torch.nn as nn
import pandas as pd

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
    
def predict(X):
    input_dim = 31  # Số lượng đặc trưng đầu vào
    model = FCNN(input_dim).to(device)
    
    # Load trọng số mô hình đã huấn luyện
    model.load_state_dict(torch.load(r'UI\Model_Rerun\fcnn\fcnn_model.pth'))
    model.eval()  # Đảm bảo mô hình ở chế độ đánh giá

    # Chuyển đổi DataFrame thành tensor
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    # Dự đoán mà không tính toán gradient
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()
    
    return y_pred[0]
