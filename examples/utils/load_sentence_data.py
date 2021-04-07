import nalp.utils.loader as l

# Loads an input .txt file with sentences
sentences = l.load_txt('data/sentence/coco_image_captions.txt').splitlines()

# Printing loaded sentences
print(sentences)
