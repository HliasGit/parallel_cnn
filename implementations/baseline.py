import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import jax
import jax.numpy as jnp
from jax import grad, vmap
import sys
from tqdm import tqdm
import jax.lax as lax
import time

seed = 42

def add_random_noise(dataset, num_corrupted_pixels, image_shape):
    # Unpack the image shape
    height, width = image_shape
    
    # Number of images
    num_images = dataset.shape[0]
    
    # Create indices for the images, heights, and widths for corrupted pixels
    image_indices = np.repeat(np.arange(num_images), num_corrupted_pixels)
    row_indices = np.random.randint(0, height, size=num_corrupted_pixels * num_images)
    col_indices = np.random.randint(0, width, size=num_corrupted_pixels * num_images)
    
    # Assign random noise (0 or 255) to the selected pixels
    dataset[image_indices, row_indices, col_indices] = np.random.choice([0, 255], size=num_corrupted_pixels * num_images)
    return dataset

def batch_convolution_2d(x, kernel):
    # Reshape inputs to match conv_general_dilated requirements
    # Add channel dimensions
    x = x[:, None, :, :]  # Shape: [batch_size, 1, height, width]
    kernel = kernel[None, None, :, :]  # Shape: [1, 1, kernel_height, kernel_width]
    
    # Perform convolution
    conv_output = lax.conv_general_dilated(
        x, 
        kernel, 
        window_strides=(1, 1),  # No stride
        padding='SAME',  # Keep input size same as output
        lhs_dilation=(1, 1),  # No input dilation
        rhs_dilation=(1, 1)   # No kernel dilation
    )
    
    # Remove added dimensions and return
    return conv_output[:, 0, :, :]

def main(n_epochs, data_folder):
    # Load the MNIST dataset
    (y_train, _), (y_test, _) = mnist.load_data()
    
    # Create a copy for noisy images
    x_train = y_train.copy()
    x_test = y_test.copy()
    
    # Add salt-and-pepper noise
    num_corrupted_pixels = 100
    x_train = add_random_noise(x_train, num_corrupted_pixels, image_shape=(28, 28))
    x_test = add_random_noise(x_test, num_corrupted_pixels, image_shape=(28, 28))
    
    print("Shape of x_train: ", x_train.shape)
    print("Shape of x_test: ", x_test.shape)
    
    # Normalize images
    y_train = y_train.astype(np.float32) / 255.0
    x_train = x_train.astype(np.float32) / 255.0
    y_test = y_test.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Define batch loss function
    def batch_loss_fn(kernel, x_batch, y_batch):
        """Compute mean loss for a batch of images."""
        y_pred = batch_convolution_2d(x_batch, kernel)
        return jnp.mean((y_pred - y_batch) ** 2)  # Mean squared error
    
    # Set random seed
    key = jax.random.PRNGKey(seed)
    
    # Initialize kernel 3x3 randomly
    kernel = jax.random.uniform(key, shape=(3, 3))
    
    # Gradient of the batch loss function w.r.t. the kernel
    batch_loss_grad = grad(batch_loss_fn)
    
    # Training loop
    learning_rate = 0.01
    losses = []
    
    # Start timing
    start_time = time.time()
    start_cpu_time = time.process_time()
    
    batch_size = 16

    for i in tqdm(range(n_epochs)):

        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]

        for j in range(0, x_train.shape[0], batch_size):
            # Get batch
            x_batch = x_train[j:j+batch_size]
            y_batch = y_train[j:j+batch_size]
            
            # Compute gradients for the batch
            gradients = batch_loss_grad(kernel, x_batch, y_batch)
            
            # Update kernel with gradient descent
            kernel -= learning_rate * gradients
        # print loss in tqdm bar
        losses.append(batch_loss_fn(kernel, x_batch, y_batch))
        tqdm.write(f"Epoch {i+1}, Loss: {losses[-1]}")
    
    # End timing
    end_time = time.time()
    end_cpu_time = time.process_time()
    
    # Calculate elapsed times
    real_time = (end_time - start_time) * 1000
    cpu_time = (end_cpu_time - start_cpu_time) * 1000
    
    # Write timings to file
    with open(f"{data_folder}/timings.txt", "w") as f:
        f.write(f"Real time:{real_time:.4f}\n")
        f.write(f"CPU time:{cpu_time:.4f}\n")
    
    # Save losses to npy file
    np.save(f"{data_folder}/losses.npy", np.array(losses))

if __name__ == "__main__":
    # Default to 100 epochs if no command-line argument is provided
    data_folder = sys.argv[1]
    n_epochs = int(sys.argv[2])
    main(n_epochs, data_folder)