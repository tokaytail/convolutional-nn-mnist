import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# transforms.Compose() works like an assembly line in which the input data
# will be processed by applying a sequence of transformations to them.
# It's good to remember that it will normally come into a PIL imagem format
# or as a NumPy array.
transform = transforms.Compose(
    [
        # transforms.ToTensor() does two things to the data:
        # 1. It will convert the image into a PyTorch tensor
        # 2. It will normalize the image pixel's values.
        #
        # Image pixels are typically in a range betwenn 0-255 ToTensor()
        # changes this range to be between 0.0-1.0.
        transforms.ToTensor(),
        # After the tensor is created transforms.Normalize() normalize it's
        # values
        # using the formula on every pixel `output = (input - mean) / std`.
        # Here, the mean and standard deviation values are set to 0.5.
        #
        # If we take a pixel with the value 1.0 (originally 255), the new value
        # becomes (1.0 - 0.5) / 0.5 = 1.0.
        # If we take a pixel with the value 0.0 (originally 0), the new value
        # becomes (0.0 - 0.5) / 0.5 = -1.0.
        #
        # So this normalization step shifts the range of your pixels values
        # from [0.0, 1.0] to [-1.0, 1.0].
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Load the training and test datasets
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# Definition of the convolutional neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # This is the first convolutional layers, it's designed to find basic
        # features like edges,
        # corners, and gradients.
        #
        # in_channels=1: It means that our neural networks expects a
        # single-channel input
        # which is appropriated for the grayscale MNIST dataset that we are
        # working on.
        #
        # out_channels=16: This is a design choice. It means the layer will
        # apply 16 different
        # filters (or kernels) to the input image. Each filter learns to detect
        # a different
        # low-level feature, resulting in 16 output "feature maps".
        #
        # kernel_size=3: Means that each of those 16 filters is a 3x3 matrix.
        #
        # padding=1: This adds a 1-pixel border around the input image. Without
        # it, a 3x3 kernel
        # would shrink the 28x28 image to 26x26. With padding=1, the output
        # dimension is preserved.
        #
        # Data transformations: [1, 28, 28] -> [16, 28, 28]
        # Notice that the depth increases to 16, one for each filter.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )

        # This is the max-pooling layer, its job is to reduce the spatial
        # dimensions (height and
        # width) of the feature maps. This makes the network more efficient and
        # helps it focus on
        # the most important features.
        #
        # kernel_size=2: It looks at feature maps in 2x2 windows.
        #
        # stride=2: It moves the window 2 pixels at a time, with no overlap.
        #
        # Data transformations: [16, 28, 28] -> [16, 14, 14]
        # It takes the maximum value from each 2x2 window, effectively halving
        # the height and width.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # This is the second convolutional layer, it takes the 16 simpler
        # feature maps from the previous layer and combines them to learn more
        # complex patterns.
        #
        # in_channels=16: This must match the out_channels from the previous
        # convolutional block (conv1 -> pool).
        #
        # out_channels=32: Now we're creating 32 new feature maps, looking for
        # even more abstract features.
        #
        # kernel_size=3, padding=1: Same logic as before, preserving the
        # spatial dimensions.
        #
        # Data transformations: [16, 14, 14] -> [32, 14, 14]
        #
        # Note: In the forward pass, you would apply the self.pool layer again
        # after this one, which would reduce
        # the shape to [32, 7, 7].
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        # This is the first fully connected, or linear, layer. It moves from
        # feature extraction to classification.
        # Before this layer, the 3D tensor must be flattened into a 1D vector.
        #
        # in_features = 32 * 7 * 7: This is the most important parameter here.
        # It's calculated from the output of
        # the last pooling layer: 32 channels, each 7x7 in size. 32 * 49 =
        # 1568. This means our flattened vector has
        # 1568 values. This number must match the flattened input.
        #
        # out_features = 128: This is a design choice. The layer transforms the
        # 1568 input features into a smaller,
        # intermediate representation of 128 features.
        #
        # Data transformation: [1568] -> [128]
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)

        # This is the final linear output layer.
        #
        # in_features = 128: This must match the out_features of the previous
        # layer (fc1).
        #
        # out_features = 10: This is determined by the problem. Since MNIST has
        # 10 possible classes (the digits 0
        # through 9), we need 10 outputs. Each output will represent the
        # model's score (or "logit") for one of the
        # digits.
        #
        # Data transformation: [128] -> [10]
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # FIRST CONVOLUTIONAL BLOCK
        # This line will perform a sequence of three operations:
        #
        # 1. self.conv1(x): The input tensor x is passed through the first
        # convolutional layer (conv1). As we saw
        # before, this changes the shape from [batch_size, 1, 28, 28] to
        # [batch_size, 16, 28, 28].
        #
        # 2. F.relu(...): The ReLU (Rectified Linear Unit) activation function
        # is applied to every element in the tensor. It's a simple function
        # that replaces all negative values with zero (f(x) = max(0, x)). This
        # introduces non-linearity into the model, allowing it to learn more
        # complex patterns. This operation does not change the shape of the
        # tensor.
        #
        # 3. self.pool(...): The result is then passed through the max-pooling
        # layer, which reduces the height and width by half.
        #
        # Data transformations: [batch_size, 1, 28, 28] -> [batch_size, 16, 14,
        # 14]
        x = self.pool(F.relu(self.conv1(x)))

        # SECOND CONVOLUTIONAL BLOCK
        # This line will also perform a sequence of three operations:
        #
        # 1. self.conv2(x): The tensor from the previous step is passed through
        # conv2, increasing the number of channels from 16 to 32.
        #
        # 2. F.relu(...): ReLU is applied again.
        #
        # 3. self.pool(...): The same pooling layer is used a second time to
        # downsample the feature maps again.
        #
        # Data transformations: [batch_size, 16, 14, 14] -> [batch_size, 32, 7,
        # 7]
        x = self.pool(F.relu(self.conv2(x)))

        # FLATTENING THE TENSOR
        # The convolutional part of the network is done. To feed this data into
        # the dense (fully connected) layers, we must flatten the 3D tensor
        # ([32, 7, 7]) into a 1D vector.
        #
        # .view() is PyTorch's method for reshaping tensors.
        #
        # The -1 is a convenient placeholder that tells PyTorch, "figure out
        # the correct batch size for this dimension automatically."
        #
        # 32 * 7 * 7 calculates the total number of features in the tensor,
        # which is, again, 1568.
        #
        # Data transformations: [batch_size, 32, 7, 7] -> [batch_size, 1568]
        x = x.view(-1, 32 * 7 * 7)

        # FIRST FULLY CONNECTED LAYER
        #
        # self.fc1(x): The layer transforms the 1568 input features into 128
        # output features.
        #
        # F.relu(...): ReLU is applied to the result.
        #
        # Data transformations: [batch_size, 1568] -> [batch_size, 128]
        x = F.relu(self.fc1(x))

        # OUTPUT LAYER
        #
        # Finally, the tensor is passed through the second and final linear
        # layer. This layer maps the 128 features to the 10 output classes,
        # producing the final raw scores (logits) for each class. No
        # activation function is applied here, as the loss function we'll
        # use (CrossEntropyLoss) expects raw logits.
        x = self.fc2(x)

        # The method returns the final tensor of shape [batch_size, 10], where each row contains the 10 scores for a single image in the batch. The highest score corresponds to the model's prediction for that image.
        return x
