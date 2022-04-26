import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=False)

        # self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(1), self.hidden_size)).to(device)

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(1), self.hidden_size)).to(device)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x.to(device), (h_0, c_0))

        # h_out = h_out.view(-1, self.hidden_size)

        # out = self.fc(h_out)

        return ula.to(device)



gpu_ids = []
if torch.cuda.is_available():
    gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
    device = torch.device(f'cuda:{gpu_ids[0]}')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

print(gpu_ids)
print("IDS")

trainX = torch.from_numpy(torch.load('train_embeddings_np_132.pth'))
trainY = torch.from_numpy(torch.load('Y_train_reshaped_np.pth'))

print(trainX.shape)
print(trainY.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(0)
# device = "cuda:0"
print("USING DEVICE", device)
dtype = torch.float

trainY_reshape = torch.reshape(trainY, (133, 960, 10 * 13 * 13 * 13)).to(device)
trainX_reshape = torch.reshape(trainX, (133, 960, 4 * 8)).to(device)

print("all data loaded")

num_epochs = 100
learning_rate = 0.01
input_size = 32 # Modify to embedding size
hidden_size = 10 * 13 * 13 * 13 # Modify to output size
num_layers = 1

lstm = LSTM(input_size, hidden_size, num_layers).to(device)

import torch.backends.cudnn as cudnn
# lstm = torch.nn.DataParallel(lstm)
# cudnn.benchmark = True

criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    print("epoch 1 starting...")

    print("trainY:", trainY_reshape.get_device())
    print("trainX:", trainX_reshape.get_device())

    outputs = lstm(trainX_reshape)

    print("outputs:", outputs.get_device())

    optimizer.zero_grad()
    print("finished optimizing...")

    # obtain the loss function
    loss = criterion(outputs, trainY_reshape)
    print("finished loss...")

    loss.backward()

    optimizer.step()
    if epoch % 1 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


print("RUNNING EVAL .........")

lstm.eval()
data_predict = lstm(trainX)
predictions = data_predict.data.numpy().flatten()
testY = trainY.flatten()


print("DOING FINAL EVAL")

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(testY, predictions), 2))
print("Mean squared error =", round(sm.mean_squared_error(testY, predictions), 2))
print("Median absolute error =", round(sm.median_absolute_error(testY, predictions), 2))
print("Explain variance score =", round(sm.explained_variance_score(testY, predictions), 2))
print("R2 score =", round(sm.r2_score(testY, predictions), 2))

np.save("predictions", predictions)

print("ALL DONE")
