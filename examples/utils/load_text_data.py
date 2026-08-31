from pathlib import Path

# Loads an input .txt file
text = Path("data/text/chapter1_harry.txt").read_text(encoding="utf-8")

# Printing loaded text
print(text)
