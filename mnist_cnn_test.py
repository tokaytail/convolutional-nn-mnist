import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import random
from mnist_cnn import Net

matplotlib.use("TkAgg")


def run_interactive_test():
    # Instatiate the network
    net = Net()

    # Load the weights saved after the training
    net.load_state_dict(torch.load("mnist_cnn.pth"))

    # Set the model to evaluation mode
    net.eval()

    # Define the same transformations you used for testing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    num_images = 10
    indices = random.sample(range(len(test_dataset)), num_images)

    print(f"Starting interactive session with {num_images} random images...")
    print("=" * 30)

    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]

        with torch.no_grad():
            image_batch = image.unsqueeze(0)
            outputs = net(image_batch)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = predicted.item()

        print(f"\n--- Image {i + 1}/{num_images} ---")
        print(f"✅ Actual Label:    {label}")
        print(f"🤖 Model Predicted: {predicted_class}")

        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Actual: {label} | Predicted: {predicted_class}")
        plt.show(block=False)  # Show plot without blocking the script

        if i <= num_images - 1:
            input("Press Enter to see the next image...")
            plt.close()


if __name__ == "__main__":
    run_interactive_test()
