"""Locale-specific expansion of plain integers before TTS phonemization."""
from __future__ import annotations

import re
import unicodedata

VI_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
EN_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
EN_TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def vi_number(number: int) -> str:
    if number < 10:
        return VI_DIGITS[number]
    if number < 20:
        return "mười" if number == 10 else "mười " + ("lăm" if number == 15 else VI_DIGITS[number - 10])
    if number < 100:
        tens, units = divmod(number, 10)
        return VI_DIGITS[tens] + " mươi" + ("" if not units else " " + ("mốt" if units == 1 else "lăm" if units == 5 else VI_DIGITS[units]))
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        return VI_DIGITS[hundreds] + " trăm" + ("" if not rest else " lẻ " + VI_DIGITS[rest] if rest < 10 else " " + vi_number(rest))
    thousands, rest = divmod(number, 1000)
    return VI_DIGITS[thousands] + " nghìn" + ("" if not rest else " lẻ " + vi_number(rest) if rest < 100 else " " + vi_number(rest))


def en_number(number: int) -> str:
    if number < 10:
        return EN_DIGITS[number]
    if number < 20:
        return EN_TEENS[number - 10]
    if number < 100:
        tens, units = divmod(number, 10)
        return EN_TENS[tens] + ("" if not units else " " + EN_DIGITS[units])
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        return EN_DIGITS[hundreds] + " hundred" + ("" if not rest else " " + en_number(rest))
    thousands, rest = divmod(number, 1000)
    return EN_DIGITS[thousands] + " thousand" + ("" if not rest else " " + en_number(rest))


def normalize_for_speech(text: str, locale: str) -> str:
    text = unicodedata.normalize("NFC", text)
    if locale not in {"vi-VN", "en-US"}:
        raise ValueError("Unsupported TTS locale")
    converter = vi_number if locale == "vi-VN" else en_number
    return re.sub(r"(?<![\w])\d{1,4}(?![\w])", lambda match: converter(int(match.group())), text)
