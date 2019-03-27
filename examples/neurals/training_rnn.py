import nalp.stream.loader as l
import nalp.stream.preprocess as p
from nalp.datasets.one_hot import OneHot
from nalp.neurals.rnn2 import RNN

# Loading a text
sentences = l.load_txt('data/chapter1_harry.txt')

# Defining a predition input
pred_input = "Mr. Dursley"

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.tokenize_to_char
)

# Applying pre-processing pipeline to sentences and pred_input
sentences = pipe(sentences)
pred_input = pipe(pred_input)

# Creating a OneHot dataset
d = OneHot(sentences, max_length=10)

# Defining a neural network based on vanilla RNN
rnn = RNN(vocab_size=d.vocab_size, hidden_size=64, learning_rate=0.001)

# Training the network
rnn.train(dataset=d, batch_size=128, epochs=10)

# # Predicting using the same input (just for checking what is has learnt)
# pred = rnn.predict(d.X, probability=False)

# # Calling decoding function to check the predictions
# # Note that if the network was predicted without probability, the decoder is also without
# pred_text = d.decode(pred[0], probability=False)
# print(''.join(pred_text))

# # Generating new text
# gen_text = rnn.generate_text(dataset=d, start_text=pred_input, length=100, temperature=0.2)
# print(''.join(gen_text))
