from torch import nn

# CNN
cnn = nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64),
    nn.MaxPool1d(kernel_size=(3), strides=2, padding=(3-1)/2),
    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64),
    nn.MaxPool1d(kernel_size=(3), strides=2, padding=(3-1)/2),
    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=64),
    nn.MaxPool1d(kernel_size=(3), strides=2, padding=(3-1)/2),
    nn.Flatten(),
    nn.Linear(in_features=1, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=1, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=1, out_features=5),
    nn.Softmax()
)