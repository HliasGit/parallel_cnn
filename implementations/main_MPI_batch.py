from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import jax
import jax.numpy as jnp
from jax import grad
from jax.scipy.signal import convolve2d

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Function to add random noise
def add_random_noise(dataset, num_corrupted_pixels, image_shape):
    height, width = image_shape
    num_images = dataset.shape[0]
    image_indices = np.repeat(np.arange(num_images), num_corrupted_pixels)
    row_indices = np.random.randint(0, height, size=num_corrupted_pixels * num_images)
    col_indices = np.random.randint(0, width, size=num_corrupted_pixels * num_images)
    dataset[image_indices, row_indices, col_indices] = np.random.choice([0, 255], size=num_corrupted_pixels * num_images)
    return dataset

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
num_corrupted_pixels = 100
image_shape = x_train.shape[1:]  # Height and width of images
learning_rate = 0.01
num_epochs = 10
batch_size = size  # Number of images per batch is the same as the number of processes

# Process 0 adds noise and normalizes the data
if rank == 0:
    x_train_noisy = add_random_noise(x_train.copy(), num_corrupted_pixels, image_shape)
    x_train_noisy = x_train_noisy.astype(np.float32) / 255.0  # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
else:
    x_train_noisy = None
    x_train = None

# Broadcast the noisy dataset and clean dataset to all processes
x_train_noisy = comm.bcast(x_train_noisy, root=0)
x_train = comm.bcast(x_train, root=0)

# Define convolution function
def convolution_2d(x, kernel):
    return convolve2d(x, kernel, mode="same", boundary="fill")

# Define loss function
def loss_fn(kernel, x, y_true):
    y_pred = convolution_2d(x, kernel)
    return jnp.mean((y_pred - y_true) ** 2)  # Mean squared error

# Gradient of the loss function w.r.t. the kernel
loss_grad = grad(loss_fn)

# Initialize shared kernel
kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])

# Training loop
num_batches = len(x_train_noisy) // batch_size

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_idx in range(num_batches):
        # Divide batch
        start = batch_idx * batch_size
        end = min(start + batch_size, len(x_train_noisy))
        batch_images = x_train_noisy[start:end]
        batch_labels = x_train[start:end]
        local_batch_size = len(batch_images)

        # Calculate local start and end indices for each process
        local_start = rank * (local_batch_size // size)
        local_end = local_start + (local_batch_size // size)

        if rank < (local_batch_size % size):
            local_start += rank
            local_end += rank + 1
        else:
            local_start += (local_batch_size % size)
            local_end += (local_batch_size % size)

        # Each process computes its local loss and gradient
        local_loss = 0.0
        local_gradient = jnp.zeros_like(kernel)

        for i in range(local_start, local_end):
            noisy_img = batch_images[i]
            clean_img = batch_labels[i]
            local_loss += loss_fn(kernel, noisy_img, clean_img)
            local_gradient += loss_grad(kernel, noisy_img, clean_img)

        # Aggregate gradients and loss across processes
        total_loss = comm.reduce(local_loss, op=MPI.SUM, root=0)
        total_gradient = comm.reduce(local_gradient, op=MPI.SUM, root=0)

        # Update kernel on root process
        if rank == 0:
            total_loss /= local_batch_size  # Normalize loss by batch size
            total_gradient /= local_batch_size  # Normalize gradients by batch size
            kernel -= learning_rate * total_gradient
            epoch_loss += total_loss

    # Print epoch loss on root
    if rank == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / num_batches:.4f}")

# Final kernel broadcasting to all processes
kernel = comm.bcast(kernel if rank == 0 else None, root=0)

# Visualization (only on root)
if rank == 0:
    denoised_images = np.array([convolution_2d(img, kernel) for img in x_train_noisy[:5]])

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train_noisy[i], cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(denoised_images[i], cmap='gray')
        plt.title("Denoised")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("denoised_images_epoch.png", dpi=300)
    print("Images saved to 'denoised_images_epoch.png'")
    plt.close()

MPI.Finalize()
