import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        # self.fc1(x): The layer transforms the 1568 input features into 128
        # output features.
        #
        # F.relu(...): ReLU is applied to the result.
        #
        # Data transformations: [batch_size, 1568] -> [batch_size, 128]
        x = F.relu(self.fc1(x))

        # OUTPUT LAYER
        # Finally, the tensor is passed through the second and final linear
        # layer. This layer maps the 128 features to the 10 output classes,
        # producing the final raw scores (logits) for each class. No
        # activation function is applied here, as the loss function we'll
        # use (CrossEntropyLoss) expects raw logits.
        x = self.fc2(x)

        # The method returns the final tensor of shape [batch_size, 10], where
        # each row contains the 10 scores for a single image in the batch. The
        # highest score corresponds to the model's prediction for that image.
        return x


net = Net()

# TRAINING THE CNN MODEL
# criterion = nn.CrossEntropyLoss(): This defines the loss function. Its job is
# to measure how far off your model's prediction is from the actual correct
# label. CrossEntropyLoss is the standard choice for multi-class classification
# problems like MNIST. It takes the model's raw output scores (logits) and the
# correct labels and computes a single number representing the error or "loss."
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(...): This defines the optimizer. Its job is to update
# the model's weights and biases to reduce the loss.
#
# optim.SGD: Here we are using Stochastic Gradient Descent, a classic and
# effective optimization algorithm.
#
# net.parameters(): This tells the optimizer which values it is allowed to
# modify—all the learnable parameters of your network.
#
# lr=0.001: The learning rate. This is one of the most important
# hyperparameters. It controls how large the steps are that the optimizer takes
# to update the weights. Too large, and it might overshoot the best solution;
# too small, and training will be very slow.
#
# momentum=0.9: Momentum helps the optimizer accelerate in the correct
# direction and overcome small local minima, often leading to faster and better
# training. Think of a ball rolling down a hill; it builds up momentum and
# doesn't get stuck in small divots.
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# TRAINING LOOPS
# The Outer Loop (Epochs): An epoch is one complete pass through the entire
# training dataset. The code for epoch in range(5): means you will loop over
# all the training data a total of 5 times.
for epoch in range(5):
    running_loss = 0.0

    # The Inner Loop (Batches): It's computationally expensive to process the
    # whole dataset at once. So, the trainloader breaks the data into smaller
    # batches. This inner loop iterates over each batch until it has seen all
    # the data in the dataset.
    for i, data in enumerate(trainloader, 0):
        # inputs, labels = data simply unpacks each batch
        # into the images and their corresponding correct labels. The variable
        # data has the form of [image, label] here.
        inputs, labels = data

        # optimizer.zero_grad(): RESET THE GRADIENTS. By default, PyTorch
        # accumulates gradients. This line is crucial because it clears the old
        # gradients from the previous batch. If you didn't do this, you'd be
        # accumulating gradients from all the batches, which would corrupt the
        # learning process.
        optimizer.zero_grad()

        # outputs = net(inputs): FORWARD PASS. The batch of inputs (images) is
        # passed through the network, which performs all the calculations
        # defined in your forward() method and produces a tensor of output
        # scores.
        outputs = net(inputs)

        # loss = criterion(outputs, labels): CALCULATE LOSS. The criterion
        # (CrossEntropyLoss) compares the network's outputs with the
        # ground-truth labels to calculate how wrong the network was for this
        # specific batch.
        loss = criterion(outputs, labels)

        # loss.backward(): BACKWARD PASS. This is the most magical step.
        # PyTorch's autograd engine calculates the gradient of the loss with
        # respect to every single learnable parameter in our network. These
        # gradients tell the optimizer how to adjust each parameter to reduce
        # the loss.
        loss.backward()

        # optimizer.step(): UPDATE WEIGHTS. The optimizer uses the gradients
        # computed in the backward pass to update all the network's weights and
        # biases, taking a small step in the direction that will minimize the
        # loss.
        optimizer.step()

        # LOGGING THE PROCESS
        # This code accumulates the loss for 200 batches and then prints the
        # average, giving you a real-time update on how well the model is
        # learning. It then resets the running_loss to start counting for the
        # next 200 batches.
        running_loss += loss.item()
        if i % 200 == 199:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
            running_loss = 0.0

print("Finished training")

# Define the path to save the model
PATH = "./mnist_cnn.pth"

# How it works?
# net.state_dict(): This function returns an ordered dictionary that maps each
# layer of your model to its learned parameters (weights and biases). It
# contains all the "knowledge" your model has gained during training.
#
# torch.save(object, PATH): This function takes the object (in this case, the
# state_dict dictionary) and saves it to your disk using Python's pickle
# utility. The common convention for the file extension is .pth or .pt.
#
# How to load the model?
# 1. Instantiate your model architecture:
# net = Net()
#
# 2. Load the state_dict from the file:
# net.load_state_dict(torch.load(PATH))
#
# 3. Set the model to evaluation mode (this is important for consistent results # during inference):
# net.eval()
torch.save(net.state_dict(), PATH)

print(f"Model saved to {PATH}")

correct = 0
total = 0

# DISABLING GRADIENT CALCULATION
# This is a context manager that tells PyTorch not to calculate gradients for
# any of the operations inside this block. This is crucial for two reasons:
# - Efficiency: Calculating gradients is computationally intensive and
# completely unnecessary during evaluation (also called inference). This makes
# the code run significantly faster and use less memory.
# - Correctness: It ensures that you are only testing the model. Without it,
# you could accidentally alter the model's learned weights.
with torch.no_grad():
    # This loop iterates through the testloader, which serves up batches of
    # images and their corresponding correct labels from the test set. This
    # part is identical to how the training loop gets its data.
    for data in testloader:
        images, labels = data

        # MAKING PREDICTIONS
        # This is where the model's predictions are generated and interpreted.
        #
        # outputs = net(images): This is the forward pass. The batch of test
        # images is fed into the network (net), which produces a tensor of raw
        # output scores (logits). The shape of outputs will be [batch_size, 10].
        #
        # _, predicted = torch.max(outputs.ata, 1): This line finds the most
        # likely class for each image.
        #
        # torch.max is a function that finds the maximum value along a
        # specified dimension of a tensor. We give it outputs.data along dim=1
        # (the dimension corresponding to the 10 classes). This means for each
        # image, it finds the class with the highest score.
        # torch.max returns two things: the maximum value (the score itself)
        # and the index of that value. The index corresponds to the predicted
        # class (e.g., an index of 7 means the model predicted the digit '7').
        # We only care about the index, so we assign the actual value to a
        # throwaway variable _ and store the index in predicted.
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        # COUNTING CORRECT PREDICTIONS
        # total += labels.size(0): labels.size(0) gives the number of labels in
        # the current batch (i.e., the batch size). This is added to the total
        # count.
        total += labels.size(0)

        # correct += (predicted == labels).sum().item(): This is a clever
        # one-liner.
        #
        # (predicted == labels): This compares the tensor of predictions with
        # the tensor of true labels element-wise. It produces a boolean tensor
        # like [True, False, True, True, ...], where True marks a correct
        # prediction.
        #
        # .sum(): When you sum a boolean tensor, True is treated as 1 and False # as 0. This effectively counts the number of True values, giving us
        # the number of correct predictions in the batch.
        #
        # .item(): The result of .sum() is a PyTorch tensor containing a single
        # number. .item() extracts this number as a standard Python integer,
        # which can then be added to our correct counter.
        correct += (predicted == labels).sum().item()

# CALCULATING FINAL ACCURACY
# After the loop has processed all the test batches, this line calculates and
# prints the final accuracy using the standard formula:
# Accuracy = Total Correct Predictions / Total Predictions * 100
print(f"Accuracy of the CNN on the 10000 test images: {100 * correct / total}%")
