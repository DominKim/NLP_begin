import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):

    def __init__(
        self,
        # 28
        input_size,
        hidden_size,
        output_size,
        # # Time stap일떄 만 gradient vanushing 해결이기 때문에 쌓여 있는 레이어 지정
        n_layers=4,
        # Time stap일떄 만 gradient vanushing 해결, rnn에는 batchnormalize를 안쓰기 때문에 쓴다.
        dropout_p=.2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            # 입력이 한꺼번에 주어지기 때문에 True
            bidirectional=True,
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, h, w)

        z, _ = self.rnn(x)
        # |z| = (batch_size, h, hidden_size * 2)
        # 마지막 Time step만 가져오기 때문에 -1
        z = z[:, -1]
        # |z| = (batch_size, hidden_size * 2)
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y
