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

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Divide dataset among processes
num_images = len(x_train)
images_per_process = num_images // size
#print(images_per_process)
start = rank * images_per_process
end = start + images_per_process if rank != size - 1 else num_images
#print(end)

x_train = x_train[start:end]
y_true = x_train.copy()

# Add salt-and-pepper noise to images
def add_noise(image, num_corrupted_pixels=100):
    noisy_image = image.copy()
    for _ in range(num_corrupted_pixels):
        i, j = np.random.randint(0, noisy_image.shape[0]), np.random.randint(0, noisy_image.shape[1])
        noisy_image[i, j] = np.random.choice([0, 255]) # Normalize between 0 and 1
    
    noisy_image = noisy_image.astype(np.float32) / 255.0
    return noisy_image

noisy_images = np.array([add_noise(img) for img in x_train])
# Normalize images
y_true = y_true.astype(np.float32) / 255.0

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
                    [0.0, 0.0, 0.0]])  # Random kernel for horizontal edge detection

# Training loop
learning_rate = 0.01
num_epochs = 10

for epoch in range(num_epochs):
    local_loss = 0.0
    local_gradient = 0.0

    # Each process computes loss and gradient for its images
    for i, noisy_img in enumerate(noisy_images):
        clean_img = x_train[i]
        local_loss += loss_fn(kernel, noisy_img, clean_img)
        local_gradient += loss_grad(kernel, noisy_img, clean_img)

    # Average loss and gradient across processes
    total_loss = comm.reduce(local_loss, op=MPI.SUM, root=0)
    total_gradient = comm.reduce(local_gradient, op=MPI.SUM, root=0)

    # Root process updates the kernel
    if rank == 0:
        total_loss /= num_images
        total_gradient /= num_images
        kernel -= learning_rate * total_gradient
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# Gather noisy images to rank 0 for visualization
all_noisy_images = comm.gather(noisy_images, root=0)

# Visualize and save results on rank 0
if rank == 0:
    # Flatten the list of noisy images from all processes
    all_noisy_images = np.concatenate(all_noisy_images, axis=0)

    plt.figure(figsize=(12, 6))

    # Display noisy and denoised images
    for i in range(min(5, len(all_noisy_images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(all_noisy_images[i], cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        denoised_img = convolution_2d(all_noisy_images[i], kernel)
        plt.subplot(2, 5, i + 6)
        plt.imshow(denoised_img, cmap='gray')
        plt.title("Denoised")
        plt.axis('off')

    plt.tight_layout()

    # Save the figure instead of displaying it
    plt.savefig("denoised_images.png", dpi=300)  # Save the image as a PNG file
    print("Images saved to 'denoised_images.png'")
    plt.close()  # Close the plot to free memory

MPI.Finalize()
