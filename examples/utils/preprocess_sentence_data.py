import nalp.utils.loader as l
import nalp.utils.preprocess as p

# Loads an input .txt file with sentences
sentences = l.load_txt('data/sentence/coco_image_captions.txt').splitlines()

# Creates a character pre-processing pipeline
char_pipe = p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_char)

# Creates a word pre-processing pipeline
word_pipe = p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_word)

# Applying character pre-processing pipeline to sentences
chars_tokens = [char_pipe(sentence) for sentence in sentences]

# Applying word pre-processing pipeline to sentences
words_tokens = [word_pipe(sentence) for sentence in sentences]

# Printing tokenized characters and words
print(chars_tokens)
print(words_tokens)
