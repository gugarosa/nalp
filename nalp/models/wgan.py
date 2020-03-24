import tensorflow as tf

import nalp.utils.logging as l
from nalp.core.model import Adversarial
from nalp.models.discriminators.conv import ConvDiscriminator
from nalp.models.generators.conv import ConvGenerator

logger = l.get_logger(__name__)


class WGAN(Adversarial):
    """A WGAN class is the one in charge of Wasserstein Generative Adversarial Networks implementation.

    References:
        M. Arjovsky, S. Chintala, L. Bottou. Wasserstein gan. Preprint arXiv:1701.07875 (2017).

    """

    def __init__(self, input_shape=(28, 28, 1), noise_dim=100, n_samplings=3, alpha=0.3, clip=0.01, dropout_rate=0.3):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the Generator.
            noise_dim (int): Amount of noise dimensions for the Generator.
            n_samplings (int): Number of down/up samplings to perform.
            alpha (float): LeakyReLU activation threshold.
            clip (float): Clipping value for the Lipschitz constrain.
            dropout_rate (float): Dropout activation rate.

        """

        logger.info('Overriding class: Adversarial -> WGAN.')

        # Creating the discriminator network
        D = ConvDiscriminator(n_samplings, alpha, dropout_rate)

        # Creating the generator network
        G = ConvGenerator(input_shape, noise_dim, n_samplings, alpha)

        # Overrides its parent class with any custom arguments if needed
        super(WGAN, self).__init__(D, G, name='wgan')

        # Defining the clipping value as a property for further usage
        self.clip = clip

        logger.info(
            f'Input: {input_shape} | Noise: {noise_dim} | Number of Samplings: {n_samplings} | Activation Rate: {alpha} | Clip: {clip} | Dropout Rate: {dropout_rate}.')

    @tf.function
    def D_step(self, x):
        """Performs a single batch optimization step over the discriminator.

        Args:
            x (tf.Tensor): A tensor containing the inputs.

        """

        # Defines a random noise signal as the generator's input
        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Generates new data, e.g., G(z)
            x_fake = self.G(z)

            # Samples fake targets from the discriminator, e.g., D(G(z))
            y_fake = self.D(x_fake)

            # Samples real targets from the discriminator, e.g., D(x)
            y = self.D(x)

            # Calculates the discriminator loss upon D(x) and D(G(z))
            D_loss = tf.reduce_mean(y) - tf.reduce_mean(y_fake)

        # Calculate the gradients based on discriminator's loss for each training variable
        D_gradients = tape.gradient(D_loss, self.D.trainable_variables)

        # Clips the gradients
        D_gradients = [tf.clip_by_value(
            g, -self.clip, self.clip) for g in D_gradients]

        # Applies the discriminator's gradients using an optimizer
        self.D_optimizer.apply_gradients(
            zip(D_gradients, self.D.trainable_variables))

        # Updates the discriminator's loss state
        self.D_loss.update_state(D_loss)

    @tf.function
    def G_step(self, x):
        """Performs a single batch optimization step over the generator.

        Args:
            x (tf.Tensor): A tensor containing the inputs.

        """

        # Defines a random noise signal as the generator's input
        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Generates new data, e.g., G(z)
            x_fake = self.G(z)

            # Samples fake targets from the discriminator, e.g., D(G(z))
            y_fake = self.D(x_fake)

            # Calculates the generator loss upon D(G(z))
            G_loss = -tf.reduce_mean(y_fake)

        # Calculate the gradients based on generator's loss for each training variable
        G_gradients = tape.gradient(G_loss, self.G.trainable_variables)

        # Applies the generator's gradients using an optimizer
        self.G_optimizer.apply_gradients(
            zip(G_gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(G_loss)

    def fit(self, batches, epochs=100, critic_steps=5):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing samples.
            epochs (int): The maximum number of training epochs.
            critic_steps (int): Amount of discriminator epochs per training epoch.

        """

        logger.info('Fitting model ...')

        # Iterate through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resetting states to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Iterate through all possible training batches
            for batch in batches:
                # Iterate through all possible critic steps
                for _ in range(critic_steps):
                    # Performs the optimization step over the discriminator
                    self.D_step(batch)

                # Performs the optimization step over the generator
                self.G_step(batch)

            logger.info(
                f'Loss(G): {self.G_loss.result().numpy()} | Loss(D): {self.D_loss.result().numpy()}')
