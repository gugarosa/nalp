# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from pathlib import Path

sentences = Path("data/sentence/coco_image_captions.txt").read_text(encoding="utf-8").splitlines()

print(sentences)
