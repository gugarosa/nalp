import tensorflow as tf
from tensorflow.keras import Model

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class AdversarialWrapper(Model):
    """

    """

    def __init__(self, discriminator, generator, name):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(AdversarialWrapper, self).__init__(name=name)

        #
        self.D = discriminator

        #
        self.G = generator

    def compile(self, optimizer, loss, metrics=[]):
        """Main building method.

        Args:
            optimizer (tf.optimizers): An optimizer instance.
            loss (tf.loss): A loss instance.
            metrics (list): A list of metrics to be displayed.

        """

        # Creates an optimizer object
        self.D_optimizer = optimizer

        #
        self.G_optimizer = optimizer

        # Defining the loss function
        self.loss = loss

        # Defining training accuracy metric
        self.train_accuracy = tf.metrics.CategoricalAccuracy(
            name='train_accuracy')

        # Defining training loss metric
        self.train_loss = tf.metrics.Mean(name='train_loss')

        # Defining validation accuracy metric
        self.val_accuracy = tf.metrics.CategoricalAccuracy(
            name='val_accuracy')

        # Defining validation loss metric
        self.val_loss = tf.metrics.Mean(name='val_loss')

    @tf.function
    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def discriminator_loss(true_y, fake_y):
        true_loss = self.loss(tf.ones_like(true_y), true_y)
        fake_loss = self.loss(tf.zeros_like(fake_y), fake_y)

        return true_loss + fake_loss

    def generator_loss(fake_y):
        return self.loss(tf.ones.like(fake_y), fake_y)

    @tf.function
    def step(self, x):
        """Performs a single batch optimization step.

        Args:
            x (tf.Tensor): A tensor containing the inputs.
            y (tf.Tensor): A tensor containing the inputs' labels.

        """

        # noise = tf.random.normal

        # Using tensorflow's gradient
        with tf.GradientTape() as D_tape, tf.GradientTape() as G_tape:
            #
            G_preds = self.G(noise)

            #
            true_y = self.D(x)

            #
            fake_y = self.D(G_preds)

            #
            G_loss = self.generator_loss(fake_y)

            #
            D_loss = self.discriminator_loss(true_y, fake_y)


        # Calculate the gradient based on loss for each training variable
        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)

        #
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        # Apply gradients using an optimizer
        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))

        #
        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))

    def fit(self, batches, val_batches=None, epochs=100):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing (x, y) pairs.
            val_batches (Dataset): Validation batches containing (x, y) pairs.
            epochs (int): The maximum number of training epochs.

        """

        logger.info('Fitting model ...')

        # Iterate through all epochs
        for epoch in range(epochs):
            # Iterate through all possible training batches, dependending on batch size
            for x_batch, y_batch in batches:
                # Performs the optimization step
                self.step(x_batch, y_batch)

            logger.debug(
                f'Epoch: {epoch+1}/{epochs} | Loss: {self.train_loss.result().numpy():.4f} | Accuracy: {self.train_accuracy.result().numpy():.4f}')

    def evaluate(self, val_batches):
        """Evaluates the model.

        Args:
            val_batches (Dataset): Validation batches containing (x, y) pairs.

        """

        logger.info('Evaluating model ...')

        # Resetting states to further append losses and accuracies
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

        # Iterate through all possible batches, dependending on batch size
        for x_batch, y_batch in val_batches:
            # Evaluates the network
            self.val_step(x_batch, y_batch)

        logger.info(
            f'Loss: {self.val_loss.result().numpy():.4f} | Accuracy: {self.val_accuracy.result().numpy():.4f}')

    @tf.function
    def predict(self, x):
        """Uses the model and makes a forward pass (prediction) in new data.

        Args:
            x (np.array | tf.Tensor): Can either be a numpy array or a tensorflow tensor.

        Returns:
            A tensorflow array containing the predictions. Note that if you use a softmax class in your model,
            these will be probabilities.

        """

        logger.info('Predicting with the model ...')

        # Performs the forward pass
        preds = self(x)

        return preds