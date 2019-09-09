import nalp.utils.loader as l
import nalp.utils.preprocess as p

# Loads an input .txt file
text = l.load_txt('data/text/chapter1_harry.txt')

# Creates a character pre-processing pipeline
char_pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_char
)

# Creates a word pre-processing pipeline
word_pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_word
)

# Applying character pre-processing pipeline to text
chars = char_pipe(text)

# Applying word pre-processing pipeline to text
words = word_pipe(text)

# Printing tokenized characters and words
print(chars)
print(words)