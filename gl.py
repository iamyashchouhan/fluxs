import torch
import torch.nn as nn
import torch.optim as optim
import time


class LargeDummyModel(nn.Module):
    def __init__(self):
        super(LargeDummyModel, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return x


model = LargeDummyModel().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


dummy_input = torch.randn(128, 4096).cuda()
dummy_target = torch.randn(128, 4096).cuda()


start_time = time.time()
while time.time() - start_time < 10:
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()

torch.cuda.empty_cache()
print("completed.")
