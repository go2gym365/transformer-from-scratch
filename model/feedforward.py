import torch
import torch.nn as nn
import torch.nn.functional as F

# d_model = 512
# d_ff = 2048

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))


def main():
    d_model = 2
    d_ff = 3
    dropout = 0.1
    
    pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
    print(pwff)
    print("w1 bias: ", pwff.w1.bias)
    print("w1 weight: ", pwff.w1.weight)
    return

if __name__ == "__main__":
    main()