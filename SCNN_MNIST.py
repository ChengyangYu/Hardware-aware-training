import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
learning_rate = 0.01
num_epochs = 200
noise_factor = 0.1
T = 1

out_dir = './' + time.strftime('%m-%d_%H.%M', time.localtime()) + str(batch_size) + "_" + \
          str(learning_rate) + "_" + str(num_epochs) + "_" + str(noise_factor) + "_" + str(T)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print(f'Mkdir {out_dir} ')

binary = transforms.Lambda(lambda x: 1.0 * (x > 0))
transform = transforms.Compose([
    transforms.ToTensor(),
    binary
])

train_dataset = datasets.MNIST(root='./', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SpikingNet_singleLayer(nn.Module):
    """USING IF NEURON TO TAKE THE PLACE OF THE THRESHOLD FUNCTION"""

    def __init__(self, T: int):
        super(SpikingNet_singleLayer, self).__init__()
        self.T = T

    def forward(self, x):
        x_seq = self.static_conv(x)
        x_seq = x_seq.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_seq = self.conv(x_seq)  # 13
        x_flatten = self.flatten(x_seq)
        x_fc = self.linear(x_flatten)
        functional.reset_net(self)
        return x_fc.mean(0)

    def positive_weight_constraint(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data.clamp_(0, 1)


class Spiking2(SpikingNet_singleLayer):
    def __init__(self, T=4, Cout=16, backend='torch', surrogate_function=surrogate.ATan()):
        super(Spiking2, self).__init__(T)
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, Cout, kernel_size=(3, 3), stride=(1, 1), bias=False),
        )
        functional.set_step_mode(self.static_conv, step_mode='m')
        self.conv = nn.Sequential(
            neuron.IFNode(backend=backend, surrogate_function=surrogate_function, step_mode='m'),
            layer.SeqToANNContainer(
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        )
        functional.set_step_mode(self.conv, step_mode='m')
        self.flatten = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Flatten(),
            ),
        )
        functional.set_step_mode(self.flatten, step_mode='m')
        self.linear = nn.Sequential(
            layer.SeqToANNContainer(torch.nn.Linear(in_features=Cout * 13 * 13, out_features=10, bias=False), ),
        )
        functional.set_step_mode(self.linear, step_mode='m')


model = Spiking2(T=T).to(device)
print("model", model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
max_acc = 0.0
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # Add noise to the inputs
        inputs = inputs + noise_factor * torch.randn_like(inputs)

        # inputs = torch.clamp(inputs, 0.0, 1.0)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        # limit the parameters
        model.positive_weight_constraint()

        # Print statistics
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # Evaluate the model on the test set
    correct = 0
    total = 0
    test_acc = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total

    if max_acc < test_acc:
        max_acc = test_acc
        checkpoint = {'model': model.state_dict()}
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_best.pth'))
    print(f'epoch={epoch}, train_loss={train_loss}, 'f'test_acc={test_acc}, max_test_acc={max_acc}')
