from pathlib import Path

# Loads an input .txt file with sentences
sentences = (
    Path("data/sentence/coco_image_captions.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)

# Printing loaded sentences
print(sentences)
