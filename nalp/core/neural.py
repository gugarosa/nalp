import nalp.utils.logging as l
import tensorflow as tf

logger = l.get_logger(__name__)


class Neural(tf.keras.Model):
    """A Neural class is responsible for holding vital information when defining a
    neural network.
    
    Note that some methods have to be redefined when using its childs.

    """

    def __init__(self):
        """Initialization method.
        
        Note that basic variables shared by all childs should be declared here.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Neural, self).__init__()

    def _build(self):
        """Main building method.

        Note we need to build the model itself (layers), its learning objects (learners)
        and finally, its metrics.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def _build_layers(self):
        """Builds the model layers itself.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def _build_learners(self):
        """Builds all learning-related objects (i.e., loss and optimizer).

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def _build_metrics(self):
        """Builds any desired metrics to be used with the model.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

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

    @tf.function
    def step(self, X_batch, Y_batch):
        """Performs a single batch optimization step.

        Args:
            X_batch (tf.Tensor): A tensor containing the inputs batch.
            Y_batch (tf.Tensor): A tensor containing the inputs' labels batch.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self(X_batch)

            # Calculate the loss
            loss = self.loss(Y_batch, preds)

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply gradients using an optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        # Update the loss metric state
        self.loss_metric.update_state(loss)

        # Update the accuracy metric state
        self.accuracy_metric.update_state(Y_batch, preds)

    def train(self, dataset, batch_size=1, epochs=100):
        """Trains a model.

        Args:
            dataset (Dataset): A Dataset object containing already encoded data (X, Y).
            batch_size (int): The maximum size for each training batch.
            epochs (int): The maximum number of training epochs.

        """

        logger.info(f'Model ready to be trained for: {epochs} epochs.')
        logger.info(f'Batch size: {batch_size}.')

        # Creating batches to further feed the network
        batches = dataset.create_batches(dataset.X, dataset.Y, batch_size)

        # Iterate through all epochs
        for epoch in range(epochs):
            # Resetting states to further append losses and accuracies
            self.loss_metric.reset_states()
            self.accuracy_metric.reset_states()

            # Iterate through all possible batches, dependending on batch size
            for X_batch, Y_batch in batches:
                # Performs the optimization step
                self.step(X_batch, Y_batch)

            logger.debug(
                f'Epoch: {epoch}/{epochs} | Loss: {self.loss_metric.result().numpy():.4f} | Accuracy: {self.accuracy_metric.result().numpy():.4f}')

    @tf.function
    def predict(self, X):
        """Uses the model and makes a forward pass (prediction) in new data.

        Args:
            X (np.array | tf.Tensor): Can either be a numpy array or a tensorflow tensor.

        Returns:
            A tensorflow array containing the predictions. Note that if you use a softmax class in your model,
            these will be probabilities.

        """

        # Performs the forward pass
        preds = self(X)

        return preds
