class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # Multilied
        self.bias = nn.Parameter(torch.zeros(features)) # Added
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdims = True)
        std = x.std(dim = -1, keepdims =True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))