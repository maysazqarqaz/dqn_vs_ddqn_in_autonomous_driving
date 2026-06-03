import torch
import torch.nn as nn


class DDQN(nn.Module):

    def __init__(self, num_frames=8, num_actions=6, lstm_hidden_size=256, num_lstm_layers=1):
        super().__init__()

        self.num_frames = num_frames

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    
        )

        cnn_out = self._cnn_output_size()  

        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _cnn_output_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 84, 84)
            return int(self.conv(dummy).numel())

    def forward(self, x):
        b, t, h, w = x.shape

        x = x.view(b * t, 1, h, w)
        x = self.conv(x)
        x = x.view(b, t, -1)         

        x, _ = self.lstm(x)
        x = x[:, -1, :]              

        return self.fc(x)            
