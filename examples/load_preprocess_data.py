import feelyng.stream.data as d
import feelyng.stream.loader as l
import feelyng.stream.preprocess as p

import pandas as pd

# Loads an input .csv
csv = l.load_csv('data/twitter_en.csv')

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char
)

# Transforming dataframe into samples and labels
X = csv['text']
Y = csv['sentiment']

# Applying pre-processing pipeline to X
X = X.apply(lambda x: pipe(x))

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = d.split_data(X, Y, 0.5)
