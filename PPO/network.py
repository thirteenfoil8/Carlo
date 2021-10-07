
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):

    def __init__(self):
        # The entry needs to be the state of the env. Somethings like the position of the car and the location of the borders for example.
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # 
            nn.Conv1d(6, 8, kernel_size=1, stride=2),
            nn.ReLU(),  # activation
            #nn.Conv1d(8, 16, kernel_size=1, stride=2),
            #nn.ReLU(),  # activation
            #nn.Conv1d(16, 32, kernel_size=1, stride=2),  
            #nn.ReLU(),  # activation
            #nn.Conv1d(32, 64, kernel_size=1, stride=2), 
            #nn.ReLU(),  # activation
            #nn.Conv1d(64, 128, kernel_size=1, stride=1), 
            #nn.ReLU(),  # activation
            #nn.Conv1d(128, 256, kernel_size=1, stride=1),
            #nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(8, 50), nn.ReLU(), nn.Linear(50, 1))
        self.fc = nn.Sequential(nn.Linear(8, 50), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(50, 2), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(50, 2), nn.Softplus())
        self.apply(self._weights_init)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x= x.view(-1,6,1)
        x = self.cnn_base(x)
        x = x.view(-1, 8)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v
