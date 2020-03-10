import nalp.utils.loader as l
import nalp.utils.preprocess as p

# Loads an input document .txt file
doc = l.load_doc('data/document/chapter1_harry.txt')

# Creates a pre-processing pipeline
pipe = p.pipeline(p.lower_case, p.valid_char)

# Applying character pre-processing pipeline to text
tokens = [pipe(sent) for sent in doc]

# Printing tokenized sentences
print(tokens)
