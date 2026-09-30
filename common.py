"""Shared helpers: hashing, binning, char-class masks."""
from __future__ import annotations

import hashlib

import config


def sha1_hex_upper(password: str) -> str:
    """SHA-1 of the UTF-8 password, uppercase hex — the form the HIBP API uses."""
    return hashlib.sha1(password.encode("utf-8")).hexdigest().upper()


def half_for_split(hash_hex: str, split_digit: int) -> str:
    """'A' (adaptation) if the chosen hex digit is 0-7, else 'B' (evaluation)."""
    return "A" if hash_hex[split_digit] in "01234567" else "B"


def length_bucket(n: int) -> int:
    """Collapse extreme lengths so bins are not sparsely populated."""
    if n <= config.LEN_MIN_BUCKET:
        return config.LEN_MIN_BUCKET
    if n >= config.LEN_MAX_BUCKET:
        return config.LEN_MAX_BUCKET
    return n


def charclass_mask(password: str) -> int:
    """4-bit mask: lower / upper / digit / symbol presence."""
    m = 0
    for ch in password:
        if ch.islower():
            m |= 1
        elif ch.isupper():
            m |= 2
        elif ch.isdigit():
            m |= 4
        else:
            m |= 8
    return m


def bin_key(length: int, zscore: int, mask: int) -> str:
    """String key for a (length-bucket, strength[, char-class]) bin."""
    lb = length_bucket(length)
    if config.USE_CHARCLASS_MASK:
        return f"{lb}|{zscore}|{mask}"
    return f"{lb}|{zscore}"
