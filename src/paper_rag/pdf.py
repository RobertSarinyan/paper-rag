"""Extract page text while preserving its source page number."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .console import configure_console_output


@dataclass(frozen=True)
class PageText:
    page_number: int  # One-based, like the page number shown to a reader.
    text: str


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Return text for every PDF page, including pages with no extractable text."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {path}")

    reader = PdfReader(path)
    pages = [
        PageText(page_number=number, text=page.extract_text() or "")
        for number, page in enumerate(reader.pages, start=1)
    ]
    if not any(page.text.strip() for page in pages):
        raise ValueError("No extractable text found; the PDF may need OCR")
    return pages


def main() -> None:
    configure_console_output()
    parser = argparse.ArgumentParser(description="Inspect text extracted from a PDF")
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
    parser.add_argument("--page", type=int, default=1, help="Page to preview (default: 1)")
    args = parser.parse_args()

    pages = extract_pages(args.pdf)
    if not 1 <= args.page <= len(pages):
        parser.error(f"--page must be between 1 and {len(pages)}")

    print(f"Pages: {len(pages)}")
    for page in pages:
        print(f"Page {page.page_number}: {len(page.text)} characters")
    print(f"\nPreview of page {args.page}:\n")
    print(pages[args.page - 1].text[:1000])


if __name__ == "__main__":
    main()
