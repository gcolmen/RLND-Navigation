import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.step_window = 5
        input_size = state_size 
        # Inputs to hidden layer linear transformation
        
        # RNN Layer
        self.hidden_dim = 128
        n_layers = 2
        self.input = nn.RNN(input_size, self.hidden_dim, n_layers, batch_first=True)
        
        self.hidden = nn.Linear(self.hidden_dim * self.step_window, 256)
        self.hidden2 = nn.Linear(256, 64)

        # Output layer, one for each possible action
        self.output = nn.Linear(64, action_size)
        
        # Define sigmoid activation and softmax output 
        self.relu = torch.nn.ReLU()
        
        #Loss function and optimizer
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.01)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Pass the input tensor through each of our operations
        x, h = self.input(state)
        #x = x[:, self.step_window - 1, :] ##batch, seq, inp_size
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #x = x.contiguous().view(-1, self.hidden_dim)
        x = x.contiguous().view(state.shape[0], -1)
        
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        
        x = self.output(x)
        
        return x
