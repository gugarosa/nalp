import tensorflow as tf

from nalp.models.adversarial.gan import GAN

# Defining a constant to hold the input noise dimension
N_NOISE = 100

# Defining a constant to hold the number of features
N_FEATURES = 784

# Defining a constant to hold the batch size
BATCH_SIZE = 256

# Loading the MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshaping the training data
x_train = x_train.reshape(x_train.shape[0], N_FEATURES).astype('float32')

# Normalizing the data
x_train = (x_train - 127.5) / 127.5

# Creating the dataset from shuffle and batched data
dataset = tf.data.Dataset.from_tensor_slices(
    x_train).shuffle(100000).batch(BATCH_SIZE)

# Creating the GAN
gan = GAN(gen_input=N_NOISE, gen_output=N_FEATURES, alpha=0.01)

# Compiling the GAN
gan.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
            loss=tf.losses.BinaryCrossentropy(from_logits=True))

# Fitting the GAN
gan.fit(dataset, epochs=150)

# Saving GAN weights
gan.save_weights('trained/gan', save_format='tf')
