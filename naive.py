import torch

class GeneratedTensorDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self):
        self.chosen_one = torch.randn(3, 3)
        self.total_size = 300

        # inputs and outputs.
        u=[x for x in map(lambda i: torch.randn(3), range(self.total_size))]
        v=[x for x in map(lambda i: self.chosen_one@torch.randn(3), range(self.total_size))]

        self.inputs = torch.stack(u, dim=0)
        self.labels = torch.stack(v, dim=0)

        # Move model and data to GPU
        print(self.chosen_one)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class NaiveNetwork(torch.nn.Module):
    def __init__(self):
        super(NaiveNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(3, 9, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(9, 9, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(9, 3, bias=False)
        )

        self.batch_size = 3
        self.learning_rate = 0.0002
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opti = torch.optim.SGD(self.parameters(), lr = self.learning_rate)

        data_set = GeneratedTensorDataSet()
        self.data_loader = torch.utils.data.DataLoader(data_set, batch_size=self.batch_size, shuffle=True)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_stack(x)

    def train_loop(self):
        for batch, (x, y) in enumerate(self.data_loader):
            pred = self(x)
            loss = self.loss_fn(pred, y)
            self.opti.zero_grad()
            loss.backward()
            self.opti.step()
            print(batch, " ", loss.item())

model = NaiveNetwork()
print(model)
print("=====================================")
model.train_loop()

# Now this is the training