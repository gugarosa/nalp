import nalp.utils.decorators as d
import nalp.utils.logging as l
import tensorflow as tf
import numpy as np
from nalp.core.neural import Neural

logger = l.get_logger(__name__)


class RNN(Neural):
    """A RNN class is the one in charge of Recurrent Neural Networks vanilla implementation.
    They were implemented from this paper:

    Methods:
        model(): tf.nn.dynamic_rnn with tf.nn.rnn_cell.BasicRNNCell
        loss(): tf.nn.softmax_cross_entropy_with_logits_v2
        optimizer(): tf.train.AdamOptimizer
        predictor(): tf.argmax

    """

    def __init__(self, max_length=1, vocab_size=1, hidden_size=2, learning_rate=0.001):
        """Initialization method.

        Args:
            max_length (int): The maximum length of the encoding.
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The amount of hidden neurons.
            learning_rate (float): A big or small addition on the optimizer steps.

        """

        logger.info('Overriding class: Neural -> RNN.')

        # Overrides its parent class with any custom arguments if needed
        super(RNN, self).__init__(max_length=max_length, vocab_size=vocab_size)

        # We need to create a property holding the max length of the encoding
        self._max_length = max_length
        # One for vocab size
        self._vocab_size = vocab_size
        # One for the amount of hidden neurons
        self._hidden_size = hidden_size
        # And the last for the learning rate
        self._learning_rate = learning_rate

        # The implemented methods should also be instanciated
        # Defines the model
        self.model
        # Calculates the loss function
        self.loss
        # Defines the accuracy function
        self.accuracy
        # Creates the optimization task
        self.optimizer
        # If you wish, predict new inputs based on indexes
        self.predictor
        # Or probabilities
        self.predictor_prob

    @property
    def max_length(self):
        """The maximum length of the encoding.

        """

        return self._max_length

    @property
    def vocab_size(self):
        """The size of the vocabulary.

        """

        return self._vocab_size

    @property
    def hidden_size(self):
        """The amount of hidden neurons.

        """

        return self._hidden_size

    @property
    def learning_rate(self):
        """A big or small addition on the optimizer steps.

        """

        return self._learning_rate

    @d.define_scope
    def model(self):
        """ The model should be constructed here. You can use whatever tensorflow
        operations you need.

        Returns:
            The model for further optimization and learning.

        """

        logger.debug(f'Constructing model with shape: ({self.hidden_size}, {self.vocab_size}).')

        # W will be the weight matrix
        self.W = tf.Variable(tf.random_normal([self.hidden_size, self.vocab_size]))

        # b will the bias vector
        self.b = tf.Variable(tf.random_normal([self.vocab_size]))

        # For vanilla RNN, we will use a basic RNN cell
        self.cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)

        # We do also need to create the RNN object itself
        self.o, self.h = tf.nn.dynamic_rnn(
            self.cell, self.x, dtype=tf.float32)

        # Transposing the vector dimensions
        self.o = tf.transpose(self.o, [1, 0, 2])

        # Gathering the reserve array
        self.o = self.o[-1]

        return tf.matmul(self.o, self.W) + self.b

    @d.define_scope
    def loss(self):
        """The loss function should be defined according your knowlodge.
        
        Returns:
            The loss function itself.
        
        """

        # Defining the loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.model, labels=self.y)

        # Applying an extra operation on defined loss
        loss = tf.reduce_mean(cross_entropy)

        logger.debug(f'Loss function: {loss}.')

        return loss

    @d.define_scope
    def accuracy(self):
        """Calculates the accuracy between predicted and true labels

        Returns:
            The accuracy value itself.
        """

        # Defining the accuracy function
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1)), tf.float32))

        logger.debug(f'Accuracy function: {accuracy}.')

        return accuracy

    @d.define_scope
    def optimizer(self):
        """An optimizer is the key of the learning task. Define your own
        as you may please.

        Returns:
            An optimizer object that minimizes the loss function.
        
        """

        # Creates an optimizer object
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        logger.debug(f'Optimizer: {optimizer} | Learning rate: {self.learning_rate}.')

        return optimizer.minimize(self.loss)

    @d.define_scope
    def predictor(self):
        """A predictor is responsible for returning an understable output.

        Returns:
            The index of the character with the highest probability of being the target.
        
        """

        return tf.cast(tf.argmax(self.model, 1), tf.int32)

    @d.define_scope
    def predictor_prob(self):
        """A predictor is responsible for returning an understable output.

        Returns:
            The probability array of an index being the target.
        """

        return tf.nn.softmax(self.model)

    def train(self, input_batch, target_batch, epochs=100, verbose=0, save_model=1):
        """Trains a model.

        Args:
            input_batch (tensor): Input data tensor [None, max_length, vocab_size].
            target_batch (tensor): Input labels tensor [None, vocab_size].
            epochs (int): The maximum number of training epochs.
            verbose (boolean): If verbose is true, additional printing will be done.
            save_model (boolean): If save_model is true, model will be saved into models/.
        
        """

        logger.info(f'Model ready to be trained for: {epochs} epochs.')

        # Initializing all tensorflow variables
        init = tf.global_variables_initializer()

        # Instanciating a new tensorflow session
        sess = tf.Session()

        # Running the first session's step
        sess.run(init)

        # Iterate through all epochs
        for epoch in range(epochs):
            # We run the session by feeding inputs to it (X and Y)
            _, loss, acc = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict={
                               self.x: input_batch, self.y: target_batch})

            # If verbose is True, additional printing will be made
            if (verbose):
                logger.debug(f'Epoch: {epoch}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.4f}')

        # If save model is True, we will save it for further restoring
        if (save_model):
            # Declaring a saver object for saving the model
            saver = tf.train.Saver()

            # Creating a custom string to be its output name
            self._model_path = f'models/rnn-hid{self.hidden_size}-lr{self.learning_rate}-e{epochs}-loss{loss:.4f}'

            # Saving the model
            saver.save(sess, self._model_path)

            logger.info(f'Model saved: {self._model_path}.')

    def predict(self, input_batch, model_path=None, probability=1):
        """Predicts a new input based on a pre-trained network.

        Args:
            input_batch (tensor): An input batch to be predicted.
            model_path (str): A string holding the path to the desired model.
            probability (boolean): If true, will return a probability insteaf of a label.

        Returns:
            The index of the prediction.

        """

        # Declaring a saver object for saving the model
        saver = tf.train.Saver()

        # Instanciating a new tensorflow session
        sess = tf.Session()

        # Restoring the model, should use the same name as the one it was saved
        if (self._model_path):
            saver.restore(sess, self._model_path)
        else:
            self._model_path = model_path
            saver.restore(sess, model_path)

        logger.info(f'Model restored from: {self._model_path}.')
        logger.info(f'Predicting with probability={probability}.')

        # Running the predictor method according to argument
        if (probability):
            predict = sess.run([self.predictor_prob], feed_dict={self.x: input_batch})
        else:
            predict = sess.run([self.predictor], feed_dict={self.x: input_batch})

        return predict

    def generate_text(self, dataset, start_text='', max_length=1, model_path=None):
        """Generates a maximum length of new text based on the probability of next char
        ocurring.

        Args:
            dataset (OneHot): A OneHot object.
            start_text (str): The initial text for generating new text.
            max_length (int): Maximum amount of generated text.
            model_path (str): If needed, will load a different model from the previously trained.

        Returns:
            A custom generated text.
        
        """

        logger.info(f'Generating new text with length: {max_length}.')

        # Declaring a saver object for saving the model
        saver = tf.train.Saver()

        # Instanciating a new tensorflow session
        sess = tf.Session()

        # Restoring the model, should use the same name as the one it was saved
        if (self._model_path):
            saver.restore(sess, self._model_path)
        else:
            saver.restore(sess, model_path)

        # Defining variable to hold decoded generation
        output_text = ''.join(start_text)

        # Creating tokens from starting text
        tokens = list(start_text)

        # Iterate through maximum desired length
        for _ in range(max_length):
            # Indexate tokens
            idx_token = dataset.indexate_tokens(tokens, dataset.vocab_index)
            
            # Creates the input batch
            x_p, _ = dataset.encode_tokens(idx_token, dataset.max_length, dataset.vocab_size)

            # Calculates the prediction
            predict = sess.run([self.predictor_prob], feed_dict={self.x: x_p})

            # Chooses a index based on the predictions probability distribution
            idx = np.random.choice(
                range(dataset.vocab_size),
                p=predict[0][-1]
            )

            # Removing first token
            del tokens[0]

            # Appending newest character as last token
            tokens.append(dataset.index_vocab[idx])
        
            # Outputting generated characters to start text
            output_text += dataset.index_vocab[idx]

        return output_text
