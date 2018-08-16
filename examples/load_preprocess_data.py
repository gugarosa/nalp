import feelyng.stream.loader as l
import feelyng.stream.preprocess as p

# Loads an input data
data = l.load_csv('data/twitter_en.csv')

# Creates a preprocessing pipeline
pipe = p.pipeline(
    p.lower_case
)

# Apply pipeline to data
data['text'] = data['text'].apply(lambda x: pipe(x))

print(data['text'])