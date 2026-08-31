import numpy as np
import pytest

from nalp.core import Corpus
from nalp.corpus import TextCorpus
from nalp.encoders import IntegerEncoder
from nalp.utils import loader
from nalp.utils import logging as nalp_logging
from nalp.utils.preprocess import (
    lower_case,
    pipeline,
    tokenize,
    tokenize_to_char,
    tokenize_to_word,
    valid_char,
)


def test_tokenize_filters_lowercases_and_selects_token_type():
    text = "Hello, [WORLD]^_` 42!\tNext"
    cleaned = "hello world 42\tnext"

    assert tokenize(text, "char") == list(cleaned)
    assert tokenize(text, "word") == ["hello", "world", "42", "next"]

    with pytest.raises(RuntimeError, match="`char` or `word`"):
        tokenize(text, "sentence")

    preprocess = pipeline(lower_case, valid_char, tokenize_to_word)
    assert preprocess(text) == ["hello", "world", "42", "next"]
    assert tokenize_to_char("ab") == ["a", "b"]


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
    source.write_bytes("Olá, WORLD!\r\n".encode())

    corpus = TextCorpus(from_file=source, corpus_type="word")

    assert corpus.tokens == ["ol", "world"]
    assert loader.load_txt(source) == "Olá, WORLD!\r\n"


def test_logging_helpers_preserve_public_api(tmp_path, monkeypatch):
    monkeypatch.setattr(nalp_logging, "LOG_FILE", str(tmp_path / "nalp.log"))
    logger = nalp_logging.get_logger("nalp.tests.public-api")
    logger.to_file("request %s", "complete", extra={"request_id": "abc"})

    assert isinstance(logger, nalp_logging.Logger)
    assert isinstance(nalp_logging.get_console_handler(), nalp_logging.StreamHandler)
    assert isinstance(
        nalp_logging.get_timed_file_handler(),
        nalp_logging.TimedRotatingFileHandler,
    )
    assert (
        (tmp_path / "nalp.log")
        .read_text(encoding="utf-8")
        .endswith("request complete\n")
    )

    logger.addFilter(lambda record: False)
    logger.to_file("filtered")
    assert "filtered" not in (tmp_path / "nalp.log").read_text(encoding="utf-8")


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
