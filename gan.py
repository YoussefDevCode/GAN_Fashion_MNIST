# import the necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


# Create the Discriminator class
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Build the Sequantial model
        self.model = keras.Sequential(
            [
                # Input layer with the specified shape
                # Adjust input shape as needed
                keras.layers.Input(shape=(28, 28, 1)),
                # First convolutional layer with 128 filters, 2x2 kernel size
                keras.layers.Conv2D(
                    128, (2, 2), strides=(2, 2), padding="same", activation="tanh"
                ),
                # Dropout layer to prevent overfitting
                keras.layers.Dropout(0.3),
                # First convolutional layer with 64 filters, 2x2 kernel size
                keras.layers.Conv2D(
                    64, (2, 2), strides=(2, 2), padding="same", activation="tanh"
                ),
                # Dropout layer to prevent overfitting
                keras.layers.Dropout(0.3),
                # Flatten layer
                keras.layers.Flatten(),
                # Fully connected layer
                keras.layers.Dense(128, activation="relu"),
                # Dropout layer to prevent overfitting
                keras.layers.Dropout(0.3),
                # Fully connected layer
                keras.layers.Dense(64, activation="relu"),
                # Dropout layer to prevent overfitting
                keras.layers.Dropout(0.3),
                # Output layer with 'sigmoid' activation
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

    # Define the call method for the forward pass
    def call(self, inputs):
        return self.model(inputs)


# Define the Generator
class Generator(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = keras.Sequential(
            [
                # Input layer with the specified shape
                # Adjust input shape as needed
                keras.layers.Input(shape=latent_dim),
                # Fully connected layer
                keras.layers.Dense(6272, activation="relu"),
                # Reshpae to (7,7,128) shape
                keras.layers.Reshape((7, 7, 128)),
                # Transposed convolution layer with 128 filters, 5x5 kernel
                keras.layers.Conv2DTranspose(
                    128, (5, 5), strides=(2, 2), padding="same"
                ),
                # LeakyReLU activation Generally good for GANs models
                keras.layers.LeakyReLU(alpha=0.2),
                # Dropout layer to prevent overfitting
                keras.layers.Dropout(0.3),
                # Transposed convolution layer with 128 filters, 5x5 kernel
                keras.layers.Conv2DTranspose(
                    64, (5, 5), strides=(2, 2), padding="same"
                ),
                # LeakyReLU activation
                keras.layers.LeakyReLU(alpha=0.2),
                # Dropout layer to prevent overfitting
                keras.layers.Dropout(0.3),
                # Transposed convolution layer with one filter, 3x3 kernel
                keras.layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding="same"),
            ]
        )

    # Define the call method for the forward pass
    def call(self, inputs):
        return self.model(inputs)


# Define the logical GAN model
class Gan(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(Gan, self).__init__()
        # The Generator
        self.generator = generator
        # The Discriminator
        self.discriminator = discriminator
        # Set the Discriminator weights to not trainable
        self.discriminator.trainable = False
        self.model = keras.Sequential([generator, discriminator])

    # Define the call method for the forward pass
    def call(self, inputs):
        return self.model(inputs)


# Define load_data function
def load_data():
    # this function load Fashion_MNIST from keras datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0
    return x_train


# Get the data
x_train = load_data()


# Generate real images
def generate_real_imgs(batch_size, x_train):
    # generate indices randomlly
    random_indices = np.random.randint(60000, size=int(batch_size / 2))
    # Add the gray channel
    x_train = np.expand_dims(x_train[random_indices], axis=-1)
    # Creating labels '1' for real
    labels = np.ones(int(batch_size / 2))
    return x_train, labels


# Generate fake images
def generate_fake_imgs(latent_dim, batch_size, gen):
    # Sample 100 point from latent space
    # We use only half batch for training purpose
    noise = tf.random.normal((int(batch_size / 2), latent_dim))
    labels = tf.zeros((int(batch_size / 2), 1))
    imgs = gen.predict(noise)
    return imgs, labels

# Generate plots of data
def plot_gen(nbre, sample):
    # nbre: Number of images
    # Remove the channel axis
    sample = np.squeeze(sample, axis=-1)
    for i in range(nbre):
            # define subplot
        plt.subplot(5, 5, 1 + i)
            # turn off axis
        plt.axis("off")
            # plot raw pixel data
        plt.imshow(sample[i])
    plt.show()


# Function to train the GAN
def train(gen, disc, gan, latent_dim):
    epoches = 12
    batch_size = 2056
    for epoche in range(epoches):
        for i in range(batch_size):
            X_real, y_real = generate_real_imgs(batch_size=batch_size, x_train=x_train)
            X_fake, y_fake = generate_fake_imgs(
                latent_dim=latent_dim, batch_size=batch_size, gen=gen
            )
            r_loss = disc.train_on_batch(X_real, y_real)
            f_loss = disc.train_on_batch(X_fake, y_fake)
            d_loss = 0.5 * (r_loss + f_loss)
            noise = tf.random.normal((batch_size, latent_dim))
            labels = tf.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, labels)
        # Only to monitor the discriminator loss
        d_loss = round(d_loss, 2)
        print(f"==== d_loss = {d_loss} --- g_loss = {g_loss} =====")


# Creating the Generator and the Dsicriminator
print(f"==== The Generator ====")
latent_dim = 100
gen = Generator(latent_dim)
gen.build((None, latent_dim))
gen.summary()
print(f"==== The Discriminator ====")
disc = Discriminator()
disc.build((None, 28, 28, 1))
disc.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.binary_crossentropy,
)
disc.summary()


# Build GAN model
print(f"==== GAN ====")
gan = Gan(gen, disc)
gan.build(input_shape=(None, latent_dim))
gan.summary()
gan.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=keras.losses.binary_crossentropy,
)

# Train the GAN
latent_dim = 100
train(gen,disc,gan,latent_dim=latent_dim)



# Generate 25 images by the Generator and print theme 
noise = tf.random.normal((25,100))
plot_gen(25, gen.predict(noise))