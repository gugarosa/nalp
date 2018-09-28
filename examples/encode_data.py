import feelyng.stream.loader as l
import feelyng.stream.preprocess as p
from feelyng.core.encoder import Encoder

# Loads an input .csv
csv = l.load_csv('data/twitter_en.csv')

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_sentence
)

# Transforming dataframe into samples and labels
X = csv['text']
Y = csv['sentiment']

# Applying pre-processing pipeline to X
X = X.apply(lambda x: pipe(x))

# Creating an Encoder class
e = Encoder(type='word2vec')

# Calling its internal method to learn an encoding representation
e.learn(X)

# Calling its internal method to actually encoded the desired data
e.encode(X)

# Acessing encoded data
print(e.encoded_data)
