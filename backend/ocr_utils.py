"""
ocr_utils.py — Pre-processing helpers for OCR text before amount extraction.

Key fix: Indian PIN codes (6-digit) and GSTIN numbers were being picked up
as monetary amounts (e.g. Bangalore-560037 → ₹1037 instead of ₹940).
"""

import re


# ── Noise removal ────────────────────────────────────────────────────────────

def remove_non_monetary_numbers(text: str) -> str:
    """
    Strip patterns that look like numbers but are NOT monetary amounts:
      • Indian PIN codes  — 6-digit numbers (110001, 560037, …)
      • GSTIN            — 15-char alphanumeric (29BERPK5227D1ZA, …)
      • Phone numbers    — 10-digit sequences
      • Invoice / Bill numbers that appear after 'No' or '#'
    """
    # GSTIN  (e.g. 29BERPK5227D1ZA)
    text = re.sub(
        r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # Indian PIN codes — 6-digit numbers that start with a non-zero digit
    # Also handles "Bangalore-560037" style
    text = re.sub(r'(?<![₹\d])\b[1-9]\d{5}\b(?!\.\d)', '', text)

    # 10-digit phone numbers (with or without spaces/dashes)
    text = re.sub(r'\b\d{10}\b', '', text)

    # Invoice / bill / order / receipt numbers  (e.g. "Bill No: 1", "Token No.: 4")
    text = re.sub(
        r'(?:bill\s*no|invoice\s*no|order\s*no|receipt\s*no|token\s*no)[.:\s#]*\d+',
        '',
        text,
        flags=re.IGNORECASE,
    )

    return text


# ── Grand-Total extractor (primary fix) ──────────────────────────────────────

def extract_grand_total(text: str):
    """
    Try to pull the Grand Total / Sub Total directly using labelled patterns.
    Returns a float if found, else None.

    This runs BEFORE the generic amount scan so a clearly-labelled total
    always wins over any stray number that survived noise removal.
    """
    # Clean noise first
    clean = remove_non_monetary_numbers(text)

    labelled_patterns = [
        # Grand Total  940.00  /  Grand Total : 940  /  Grand Total  940
        r'grand\s*total[\s:₹Rs.]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'total\s*amount[\s:₹Rs.]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'net\s*(?:payable|total)[\s:₹Rs.]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'amount\s*payable[\s:₹Rs.]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'sub\s*total[\s:₹Rs.]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'total[\s:₹Rs.]+([0-9,]+(?:\.[0-9]{1,2})?)',
    ]

    for pattern in labelled_patterns:
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(',', ''))
                if value > 0:
                    print(f"[ocr_utils] Grand total found via pattern '{pattern}': {value}")
                    return value
            except ValueError:
                continue

    return None


# ── Vendor-name cleaner ───────────────────────────────────────────────────────

def clean_vendor_name(raw: str) -> str:
    """
    Light cleanup for vendor names that OCR garbled.
    • Removes stray digits at the start/end
    • Collapses multiple spaces
    • Title-cases the result
    """
    if not raw:
        return raw
    # Drop leading/trailing digits and punctuation
    cleaned = re.sub(r'^[\d\W]+|[\d\W]+$', '', raw).strip()
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.title() if cleaned else raw
