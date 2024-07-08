import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from model import Net
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')



def visualize_predictions(model, test_loader, num_images=4):#num_images显示几张就改成几
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_images:
                break
            image = images[1].view(1, 28, 28)
            predict = torch.argmax(model(image))

            plt.figure(figsize=(4, 4))
            plt.imshow(image.squeeze().numpy(), cmap='gray')
            plt.title(f"Prediction: {int(predict)}")
            # plt.axis('off')
            plt.show()



def main():

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    train_dataset = CustomDataset(csv_file='labels.txt', root_dir='handwritten_images/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


    test_dataset = CustomDataset(csv_file='test_labels.txt', root_dir='handwritten_images/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 30
    train(model, train_loader, criterion, optimizer, num_epochs)


    evaluate(model, test_loader)

    visualize_predictions(model, test_loader)

if __name__ == "__main__":
    main()