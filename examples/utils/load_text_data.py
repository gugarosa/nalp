# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from pathlib import Path

text = Path("data/text/chapter1_harry.txt").read_text(encoding="utf-8")

print(text)
