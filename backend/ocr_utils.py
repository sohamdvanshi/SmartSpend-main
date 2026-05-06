"""
ocr_utils.py — Pre-processing helpers for OCR text before amount extraction.

Key fix: Indian PIN codes (6-digit) and GSTIN numbers were being picked up
as monetary amounts (e.g. Bangalore-560037 → ₹1037 instead of ₹940).
"""

import re
from datetime import datetime

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
    text = re.sub(r'(?<![₹\d])\b[1-9]\d{5,6}\b(?!\.\d)', '', text)

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
        r'grand\s*\w{3,6}[}\]|]?\s*[:\s₹Rs.]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'sub\s*\w{2,5}[:\s₹Rs.]*([0-9,]+(?:\.[0-9]{1,2})?)',
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

def extract_upi_payment(text):
    """
    Detects GPay/PhonePe/Paytm UPI screenshots and extracts fields.
    Returns dict or None if not a UPI screenshot.
    """
    if not text:
        return None

    text_lower = text.lower()

    # Must have UPI signals to proceed
    upi_signals = ['upi transaction id', 'upi txn', 'transaction id', 
                   'upiintent', 'upi intent', 'google pay', 'phonepe', 
                   'paytm', 'completed', 'payu@', '@okicici', '@oksbi',
                   '@ybl', '@ibl', '@axl']
    
    if not any(signal in text_lower for signal in upi_signals):
        return None

    result = {}

    amount_patterns = [
    r'(?:₹|rs\.?|inr)\s*([0-9,]+(?:\.[0-9]{1,2})?)',   # ₹1144.05
    r'(?:to\s+\S+\s+)([0-9,]+\.[0-9]{2})',              # "To IRCTC 1144.05"
    r'^([0-9,]+\.[0-9]{2})$',                            # standalone decimal amount on its own line
    ]
    text_clean = re.sub(r'(?:icici|sbi|hdfc|axis|kotak|indus)\s+bank\s+\d{4}', '', text, flags=re.IGNORECASE)
    for pattern in amount_patterns:
     amount_match = re.search(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
     if amount_match:
        val = float(amount_match.group(1).replace(',', ''))
        if val > 10:   # skip tiny numbers and 4-digit bank account suffixes
            result['amount'] = val
            break
    
    # --- UPI TRANSACTION ID ---
    # "UPI transaction ID\n612634777086"
    txn_match = re.search(
        r'upi\s+transaction\s+id\s*[:\n\r]+\s*([A-Za-z0-9]+)',
        text, re.IGNORECASE
    )
    if txn_match:
        result['upi_txn_id'] = txn_match.group(1).strip()
    
    # Fallback: "UPI Ref No" or just a 12-digit number after "transaction"
    if 'upi_txn_id' not in result:
        txn_fallback = re.search(
            r'(?:transaction\s+id|txn\s+id|ref\s+no)[^\d]*(\d{10,15})',
            text, re.IGNORECASE
        )
        if txn_fallback:
            result['upi_txn_id'] = txn_fallback.group(1)

    # --- VENDOR (To field) ---
    # "To www.pmpml.org" or "To: www.pmpml.org"
    vendor_match = re.search(
        r'to\s*:?\s*([\w\.\-]+\.(?:org|com|in|net|co\.in))',
        text, re.IGNORECASE
    )
    if vendor_match:
        result['vendor'] = vendor_match.group(1).strip()
    else:
        # Fallback: UPI handle after "To:"
        upi_handle = re.search(
            r'to\s*:?\s*([\w\.\-]+@[\w]+)',
            text, re.IGNORECASE
        )
        if upi_handle:
            result['vendor'] = upi_handle.group(1).strip()

    # --- DATE ---
    # "6 May 2026, 7:03 pm"
    date_match = re.search(
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
        text, re.IGNORECASE
    )
    if date_match:
        try:
            parsed = datetime.strptime(date_match.group(1).strip(), '%d %B %Y')
            result['date'] = parsed.strftime('%Y-%m-%d')
        except:
            result['date'] = datetime.now().strftime('%Y-%m-%d')
    else:
        result['date'] = datetime.now().strftime('%Y-%m-%d')

    vendor_lower = result.get('vendor', '').lower()
    note_lower   = text.lower()

    if any(w in vendor_lower or w in note_lower for w in ['pmpml', 'irctc', 'railways', 'metro', 'bus', 'ola', 'uber', 'rapido']):
        result['category'] = 'Transportation'
    elif any(w in vendor_lower or w in note_lower for w in ['zomato', 'swiggy', 'food', 'restaurant', 'cafe']):
        result['category'] = 'Food & Dining'
    elif any(w in vendor_lower or w in note_lower for w in ['airtel', 'jio', 'bsnl', 'electricity', 'broadband']):
        result['category'] = 'Bills & Utilities'
    elif any(w in vendor_lower or w in note_lower for w in ['amazon', 'flipkart', 'myntra', 'swiggy instamart', 'blinkit']):
        result['category'] = 'Shopping'
    elif any(w in vendor_lower or w in note_lower for w in ['hospital', 'pharmacy', 'apollo', 'medplus']):
        result['category'] = 'Healthcare'
    else:
        result['category'] = 'Miscellaneous'    

    # Only return if we got at least amount
    return result if 'amount' in result else None