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

        """

        raise NotImplementedError

    def _build_layers(self):
        """Builds the model layers itself.

        """

        raise NotImplementedError

    def _build_learners(self):
        """Builds all learning-related objects (i.e., loss and optimizer).

        """

        raise NotImplementedError

    def _build_metrics(self):
        """Builds any desired metrics to be used with the model.

        """

        raise NotImplementedError

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        """

        raise NotImplementedError

    @tf.function
    def step(self, input_batch, target_batch):
        """
        """

        with tf.GradientTape() as tape:
            preds = self(input_batch)
            loss = self.loss(target_batch, preds)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(target_batch, preds)

    def train(self, dataset, epochs=10, batch_size=1):
        """
        """
        
        for _ in range(epochs):
            self.loss_metric.reset_states()
            self.accuracy_metric.reset_states()
            for input_batch, target_batch in dataset:
                self.step(input_batch, target_batch)
            print(self.loss_metric.result(), self.accuracy_metric.result())
