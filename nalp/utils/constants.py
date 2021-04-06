"""Constants.
"""

# Reserved tokens that are used to indicate fixed patterns,
# such as start-of-sentence, end-of-sentence, padding and unknown
SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'

# A buffer size constant defines the maximum amount of
# buffer that should be used when shuffling a dataset
BUFFER_SIZE = 100000

# A discriminator steps constant defines the maximum number
# of sampling steps that the discriminator should be trained on
D_STEPS = 3

# A epsilon constants defined a small value for avoiding
# unwanted mathematical errors, such as division by zero or log(0)
EPSILON = 1e-20
