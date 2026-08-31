import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from nalp.utils.preprocess import tokenize

ROOT = Path(__file__).parents[1]


def _load_module(name: str, *relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT.joinpath(*relative_path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_core_classes():
    saved_modules = {
        name: sys.modules.get(name) for name in ("nalp.core", "nalp.core.encoder")
    }

    try:
        corpus_module = _load_module("_test_core_corpus", "nalp", "core", "corpus.py")
        core_package = ModuleType("nalp.core")
        core_package.__path__ = []
        core_package.Corpus = corpus_module.Corpus
        sys.modules["nalp.core"] = core_package

        _load_module("nalp.core.encoder", "nalp", "core", "encoder.py")
        text_module = _load_module(
            "_test_text_corpus", "nalp", "corpus", "text.py"
        )
        integer_module = _load_module(
            "_test_integer_encoder", "nalp", "encoders", "integer.py"
        )

        return (
            corpus_module.Corpus,
            text_module.TextCorpus,
            integer_module.IntegerEncoder,
        )
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


Corpus, TextCorpus, IntegerEncoder = _load_core_classes()


def test_tokenize_filters_lowercases_and_selects_token_type():
    text = "Hello, [WORLD]^_` 42!\tNext"
    cleaned = "hello world 42\tnext"

    assert tokenize(text, "char") == list(cleaned)
    assert tokenize(text, "word") == ["hello", "world", "42", "next"]

    with pytest.raises(RuntimeError, match="`char` or `word`"):
        tokenize(text, "sentence")


def test_corpus_frequency_builds_direct_public_attributes():
    tokens = ["beta", "alpha", "beta", "gamma"]
    corpus = Corpus(min_frequency=2)
    corpus.tokens = tokens

    corpus._check_token_frequency()
    corpus._build()

    assert tokens == ["beta", "<UNK>", "beta", "<UNK>"]
    assert corpus.vocab == ["<UNK>", "beta"]
    assert corpus.vocab_size == 2
    assert corpus.vocab_index == {"<UNK>": 0, "beta": 1}
    assert corpus.index_vocab == {0: "<UNK>", 1: "beta"}
    assert "_tokens" not in vars(corpus)


def test_text_corpus_reads_utf8_with_pathlib(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Olá, WORLD!\n", encoding="utf-8")

    corpus = TextCorpus(from_file=source, corpus_type="word")

    assert corpus.tokens == ["ol", "world"]


def test_integer_encoder_handles_unknown_and_nested_tokens():
    encoder = IntegerEncoder()

    with pytest.raises(RuntimeError, match=r"learn\(\) prior to encode"):
        encoder.encode(["known"])
    with pytest.raises(RuntimeError, match=r"learn\(\) prior to decode"):
        encoder.decode(np.array([0]))

    dictionary = {"<UNK>": 0, "known": 1}
    reverse_dictionary = {0: "<UNK>", 1: "known"}
    encoder.learn(dictionary, reverse_dictionary)

    encoded = encoder.encode([["known", "missing"], ["missing", "known"]])

    np.testing.assert_array_equal(encoded, [[1, 0], [0, 1]])
    assert encoded.dtype == np.int32
    assert encoder.decode(encoded) == [
        ["known", "<UNK>"],
        ["<UNK>", "known"],
    ]
    assert encoder.encoder is dictionary
    assert encoder.decoder is reverse_dictionary
    assert "_encoder" not in vars(encoder)
    assert "_decoder" not in vars(encoder)
