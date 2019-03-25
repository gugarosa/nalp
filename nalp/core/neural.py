import nalp.utils.decorators as d
import tensorflow as tf
from tensorflow.keras import Model


class Neural:
    """A Neural class is responsible for holding vital information when defining a
    neural network. Note that some methods have to be redefined when using its childs.

    """

    def __init__(self, model):
        """Initialization method.
        
        Note that basic variables shared by all childs should be declared here.

        Args:
            shape (list): A list containing in its first position the shape of the inputs (x)
                and on its second position, the shape of the labels (y).

        """

        self.model = model

        self.optimizer = tf.optimizers.Adam(0.01)

        self.loss = tf.losses.CategoricalCrossentropy()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')



    @tf.function
    def step(self, input_batch, target_batch):
        with tf.GradientTape() as tape:
            preds = self.model(input_batch)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(target_batch, preds))
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return self.train_loss(loss), self.train_accuracy(target_batch, preds)



    def train(self, dataset, epochs=10, batch_size=1):
        data = dataset.create_batches(dataset.X, dataset.Y, batch_size)

        for _ in range(epochs):
            for input_batch, target_batch in data:
                loss, acc = self.step(input_batch, target_batch)
                print(loss, acc)
                
