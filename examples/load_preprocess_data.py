import nalp.stream.loader as l
import nalp.stream.preprocess as p

# Loads an input .csv
csv = l.load_csv('data/16k_twitter_en.csv')

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

# # Splitting data into training and testing sets
# X_train, X_test, Y_train, Y_test = d.split_data(X, Y, 0.5)