import torch
import torch.nn as nn
from . import s4

class S4Model(nn.Module):
    def __init__(self, config):
        super(S4Model, self).__init__()
        self.model_dim = config.model.model_dim
        self.state_dim = config.model.state_dim
        self.input_dim = config.data.input_dim
        self.n_classes = config.data.n_classes

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim,self.model_dim),
                                 nn.ReLU())
        self.s4_block = nn.Sequential(nn.LayerNorm(self.model_dim),
                                      s4.S4D(d_model=self.model_dim, d_state=self.state_dim, transposed=False))
        self.fc2 = nn.Linear(self.model_dim, self.n_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.s4_block(x)
        x = torch.max(x, axis=1).values

        logits = self.fc2(x)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = torch.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_hat': Y_hat, 'Y_prob': Y_prob}
        return results_dict