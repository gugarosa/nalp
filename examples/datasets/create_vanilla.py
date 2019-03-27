import nalp.stream.loader as l
import nalp.stream.preprocess as p
import numpy as np
from nalp.datasets.vanilla import Vanilla

# Loads an input .csv
csv = l.load_csv('data/16k_twitter_en.csv')

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_word
)

# Transforming dataframe into samples and labels
X = csv['text']
Y = csv['sentiment'].values

# Applying pre-processing pipeline to X
X = X.apply(lambda x: pipe(x)).values

# Creates the dataset
d = Vanilla(X, Y, categorical=True)

# Acessing properties from Vanilla class
# Note that every property can be acessed, please refer to the docs to know all of them
print(f'Vanilla -> X[0]:  {d.X[0]} | Y[0]: {d.Y[0]} | Label: {d._index_labels[np.argmax(d.Y[0])]}')
