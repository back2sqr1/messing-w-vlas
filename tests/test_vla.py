import pytest

from dodge.vla import MAX_WORDS, VOCAB, tokenize


def test_tokenize_pads_to_max_words():
    ids = tokenize("dodge red catch blue")
    assert len(ids) == MAX_WORDS
    assert [VOCAB[i] for i in ids] == ["dodge", "red", "catch", "blue"] + ["<pad>"] * (MAX_WORDS - 4)


def test_tokenize_rejects_unknown_words():
    with pytest.raises(ValueError):
        tokenize("dodge green")
