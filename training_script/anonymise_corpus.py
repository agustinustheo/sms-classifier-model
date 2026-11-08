"""Mask personal identifiers in the SMS corpus (run once, from training_script/).

    uv run --with openpyxl anonymise_corpus.py

Replaces, in every corpus file (.txt, .csv, .xlsx; external datasets are left as published):
  - phone numbers             -> "phonenumber"
  - e-mail addresses          -> "emailaddress"
  - names in shared contacts  -> "[Name]nama"
  - digit runs of 7 or more   -> "number"   (account, card, ID and order numbers)

The replacement words are the ones the preprocessing already produces, so the
models see the same kind of tokens. URLs are kept (they're mostly company links
and useful for spam detection). Names written inside ordinary messages are not
detected automatically. This masking was also applied to every past commit, including the `original` branch.
"""

import re
import sys
from pathlib import Path

CORPUS = Path("corpus/sms_corpus")

RULES = [
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)*(\.\.\.)?"), " emailaddress "),
    (re.compile(r"\[Name\][^;\]\n]*"), "[Name]nama"),
    # Indonesian mobile and landline numbers, with optional +62/62 prefix and separators.
    # Not after "<digit>." or "<digit>," so prices like Rp50.000.000 are left alone.
    (re.compile(r"(?<!\d)(?<!\d[.,])\(?(\+62|62|0)[\s.-]?\(?\d{2,5}\)?[\s.-]?\d{3,6}([\s.-]?\d{2,5})?(?![\w])"), " phonenumber "),
    (re.compile(r"(?<!\d)\d{7,}(?!\d)"), " number "),
]


def mask(text: str) -> str:
    for pattern, repl in RULES:
        text = pattern.sub(repl, text)
    return text


def main():
    total = 0
    for path in sorted(CORPUS.rglob("*")):
        if "external" in path.parts:
            continue
        if path.suffix in (".txt", ".csv"):
            old = path.read_text(encoding="utf-8")
            new = mask(old)
            n = sum(a != b for a, b in zip(old.splitlines(), new.splitlines()))
            if new != old:
                path.write_text(new, encoding="utf-8")
        elif path.suffix == ".xlsx":
            import openpyxl

            wb = openpyxl.load_workbook(path)
            n = 0
            for ws in wb.worksheets:
                for row in ws.iter_rows():
                    for cell in row:
                        if isinstance(cell.value, str):
                            new = mask(cell.value)
                            if new != cell.value:
                                cell.value = new
                                n += 1
                        elif isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool) \
                                and abs(cell.value) >= 1_000_000:  # a message that is just a 7+ digit number
                            cell.value = "number"
                            n += 1
            if n:
                wb.save(path)
        else:
            continue
        total += n
        print(f"{n:5d} masked  {path}")
    print(f"{total} rows/cells changed")


if __name__ == "__main__":
    sys.exit(main())
