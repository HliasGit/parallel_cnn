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

# Check if JAX is using GPU
from jax.lib import xla_bridge
print("JAX is using:", xla_bridge.get_backend().platform)

seed = 42

def add_random_noise(dataset, num_corrupted_pixels, image_shape):
    height, width = image_shape
    num_images = dataset.shape[0]
    image_indices = np.repeat(np.arange(num_images), num_corrupted_pixels)
    row_indices = np.random.randint(0, height, size=num_corrupted_pixels * num_images)
    col_indices = np.random.randint(0, width, size=num_corrupted_pixels * num_images)
    dataset[image_indices, row_indices, col_indices] = np.random.choice([0, 255], size=num_corrupted_pixels * num_images)
    return dataset

def batch_convolution_2d(x, kernel):
    x = x[:, None, :, :]
    kernel = kernel[None, None, :, :]
    conv_output = lax.conv_general_dilated(
        x, 
        kernel, 
        window_strides=(1, 1),
        padding='SAME',
        lhs_dilation=(1, 1),
        rhs_dilation=(1, 1)
    )
    return conv_output[:, 0, :, :]

def main(n_epochs, data_folder):
    (y_train, _), (y_test, _) = mnist.load_data()
    x_train = y_train.copy()
    x_test = y_test.copy()
    num_corrupted_pixels = 100
    x_train = add_random_noise(x_train, num_corrupted_pixels, image_shape=(28, 28))
    x_test = add_random_noise(x_test, num_corrupted_pixels, image_shape=(28, 28))
    print("Shape of x_train: ", x_train.shape)
    print("Shape of x_test: ", x_test.shape)
    y_train = y_train.astype(np.float32) / 255.0
    x_train = x_train.astype(np.float32) / 255.0
    y_test = y_test.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    def batch_loss_fn(kernel, x_batch, y_batch):
        y_pred = batch_convolution_2d(x_batch, kernel)
        return jnp.mean((y_pred - y_batch) ** 2)

    # key = jax.random.PRNGKey(seed)
    # kernel = jax.random.uniform(key, shape=(3, 3))
    kernel = jnp.array([[0.01, 0.0, 0.0],
                        [-1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0]])
    
    batch_loss_grad = grad(batch_loss_fn)
    learning_rate = 0.01
    losses = []
    start_time = time.time()
    start_cpu_time = time.process_time()
    batch_size = 128

    for i in tqdm(range(n_epochs), desc="Epochs"):

        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]
        
        for j in tqdm(range(0, x_train.shape[0], batch_size), desc="Batches", leave=False):
            x_batch = x_train[j:j+batch_size]
            y_batch = y_train[j:j+batch_size]
            gradients = batch_loss_grad(kernel, x_batch, y_batch)
            kernel -= learning_rate * gradients
        losses.append(batch_loss_fn(kernel, x_batch, y_batch))
        tqdm.write(f"Epoch {i+1}, Loss: {losses[-1]}")

    end_time = time.time()
    end_cpu_time = time.process_time()
    real_time = (end_time - start_time) * 1000
    cpu_time = (end_cpu_time - start_cpu_time) * 1000

    with open(f"{data_folder}/timings.txt", "w") as f:
        f.write(f"Real time:{real_time:.4f}\n")
        f.write(f"CPU time:{cpu_time:.4f}\n")

    # Save losses to npy file
    np.save(f"{data_folder}/losses.npy", np.array(losses))


if __name__ == "__main__":
    data_folder = sys.argv[1]
    n_epochs = int(sys.argv[2])
    main(n_epochs, data_folder)
