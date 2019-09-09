import nalp.utils.loader as l
import nalp.utils.preprocess as p
from nalp.core.dictionary import Dictionary

# Loads an input .txt file
text = l.load_txt('data/text/chapter1_harry.txt')

# Creates a character pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_char
)

# Creates a word pre-processing pipeline
# pipe = p.pipeline(
#     p.lower_case,
#     p.valid_char,
#     p.tokenize_to_word
# )

# Applying character pre-processing pipeline to text
tokens = pipe(text)

# Applying word pre-processing pipeline to text
# tokens = pipe(text)

# Creating a character Dictionary from tokens
d = Dictionary(tokens=tokens)

# Creating a word Dictionary from tokens
# word_dict = Dictionary(tokens=words)

# Accessing Dictionary properties
print(d.tokens)
print(d.vocab, d.vocab_size)
print(d.vocab_index, d.index_vocab)
