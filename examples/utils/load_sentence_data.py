from nalp.utils import loader

# Loads an input .txt file with sentences
sentences = loader.load_txt("data/sentence/coco_image_captions.txt").splitlines()

# Printing loaded sentences
print(sentences)
