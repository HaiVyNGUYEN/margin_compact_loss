import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
from archi import ResNet18
from training_utils import train_no, accuracy_evaluation


data_dir = 'dataset'
train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=False)
test_dataset  = torchvision.datasets.CIFAR10(data_dir, train=False, download=False)

train_transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                 (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                                ])

test_transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])

train_dataset.transform = train_transform

test_dataset.transform = test_transform

batch_size= 128

train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

now = datetime.now()
epochs = 200 
loss_fn = nn.CrossEntropyLoss()
model = ResNet18().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

#correct = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    _train_batch_loss , _train_batch_accuracy = train_no(train_loader, model, loss_fn, optimizer, device)
    scheduler.step()
    print('Time taken:', datetime.now()-now)

torch.save(model.state_dict(), './resnet18_sgd_softmax_loss_200_epochs')

print("Training Done!")
print(datetime.now())
print('Time taken:', datetime.now()-now)

model = ResNet18().to(device)
model.load_state_dict(torch.load('./resnet18_sgd_softmax_loss_200_epochs'))
print("Testing acurracy of model on test set")
correct = accuracy_evaluation(test_loader, model)
print(correct)