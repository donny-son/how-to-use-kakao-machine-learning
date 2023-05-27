import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def main():
    # Configs
    parser = argparse.ArgumentParser()

    parser.add_argument("--param", type=str, default="default value", help="help text")

    # Data configs
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--no_shuffle", action="store_true")

    # Model configs
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_features", type=int, default=32)

    # Trainer configs
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)

    # Initialize args
    args = parser.parse_args()

    # Parameters
    log_dir = os.path.join(args.output_dir, "log")
    summary_writer = SummaryWriter(log_dir=log_dir)
    model_path = os.path.join(args.output_dir, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Prepare dataset
    transform = transforms.ToTensor()
    MNIST_IMG_SIZE = 28
    MNIST_CHANNEL_SIZE = 1
    MNIST_NUM_CLASSES = 10
    mnist_train = datasets.MNIST(root=args.data_dir, train=True, transform=transform)

    mnist_test = datasets.MNIST(root=args.data_dir, train=False, transform=transform)

    # Build dataloader
    train_loader = DataLoader(
        dataset=mnist_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=not args.no_shuffle,
    )
    test_loader = DataLoader(
        dataset=mnist_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Initialize model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(MNIST_CHANNEL_SIZE, args.num_features, 3, padding=1, bias=True),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(
            MNIST_IMG_SIZE * MNIST_IMG_SIZE * args.num_features,
            MNIST_NUM_CLASSES,
            bias=False,
        ),
    ).to(device)

    # Define loss function & optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        avg_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # Prediction
            pred = model(x)
            loss = criterion(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if batch_idx % 100 == 0:
                print(
                    "[Epoch: {:>4}][Batch: {:>4}] step loss = {:>.9}".format(
                        epoch, batch_idx, loss.item()
                    )
                )

        avg_loss /= len(train_loader)
        summary_writer.add_scalar("loss", avg_loss, global_step=epoch)

    print("Training Finished!")

    # Test model
    correct = 0
    count = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        max_val, max_args = torch.max(pred, 1)

        correct += torch.sum(max_args == y).item()
        count += len(y)

    print("Testing Finished!")

    accuracy = correct / count
    print("Test accuracy:", accuracy)

    torch.jit.save(torch.jit.script(model), model_path)
    print("Model saved!")


if __name__ == "__main__":
    main()
