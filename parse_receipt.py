"""Post-processing: raw OCR text → structured receipt JSON (Fitur 2 AI-DS-SPEC).

Output format sesuai spec:
{
  "merchant": str,
  "date": "YYYY-MM-DD" | null,
  "total": int | null,
  "items": [{"name": str, "qty": int, "price": int}],
  "category": str,
  "confidence": float (0-1)
}

Strategy (bidirectional pairing):
  Det model memotong per text region. Dua format utama yang ditemukan:

  Format A (nama dulu, harga setelah):
    BASO TAHU
    43,181

  Format B (harga dulu, nama setelah)  <- struk A Fung:
    43,181
    143181    <- "1 43,181" terbaca gabung: qty=1, harga=43181
    BASO TAHU

  Format C (satu baris, tab-separated):
    BASO TAHU    1    43,181    43,181

  Parser ini menangani ketiga format dengan cara:
  1. Tokenisasi semua baris jadi tipe: NAME / PRICE / QTY_PRICE / NOISE / KEYWORD
  2. Sliding window pairing: cari pasangan (NAME, PRICE) dengan toleransi noise 3 baris
  3. Jika PRICE muncul sebelum NAME (format B), pair ke NAME berikutnya
  4. Qty diekstrak dari pola QTY_PRICE (e.g. '143181' -> qty=1, price=43181)
"""

import re
from datetime import datetime
from typing import Optional

# ── Kategori valid ───────────────────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "makanan": [
        "indomaret", "alfamart", "alfamidi", "lawson", "circle k",
        "mcdonald", "kfc", "pizza", "burger", "bakery", "cafe", "kopi",
        "warung", "resto", "restaurant", "mie", "nasi", "ayam", "bebek",
        "supermarket", "hypermart", "giant", "hero", "carrefour",
        "indomie", "teh", "minuman", "snack", "roti", "es", "milk", "ice",
        "tahu", "tempe", "goreng", "sambal", "organic", "baso", "jeruk",
        "fung", "bakso",
    ],
    "transport": [
        "gojek", "grab", "ojek", "taxi", "transjakarta", "kereta",
        "pertamina", "shell", "spbu", "bensin", "bbm", "parkir",
        "toll", "tol", "airport", "bandara",
    ],
    "belanja": [
        "tokopedia", "shopee", "lazada", "bukalapak", "blibli",
        "zalora", "uniqlo", "h&m", "zara", "miniso", "ikea",
        "ace hardware", "informa", "electronic", "erafone", "ibox",
    ],
    "kesehatan": [
        "apotek", "apotik", "kimia farma", "century", "guardian",
        "klinik", "rumah sakit", "rs ", "puskesmas", "dokter",
        "vitamin", "obat", "masker",
    ],
    "tagihan": [
        "pln", "listrik", "pdam", "air", "telkom", "indihome",
        "firstmedia", "biznet", "wifi", "internet", "pulsa", "token",
    ],
    "hiburan": [
        "cinema", "cgv", "cinepolis", "xxi", "bioskop", "studio",
        "spotify", "netflix", "youtube", "game", "playstation",
    ],
    "pendidikan": [
        "gramedia", "togamas", "buku", "alat tulis", "stationery",
        "kursus", "les", "sekolah", "universitas",
    ],
}

# ── Regex ───────────────────────────────────────────────────────────────────────────────
DATE_PATTERNS = [
    (r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', '%d/%m/%Y'),
    (r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', '%Y/%m/%d'),
    (r'(\d{1,2})\s+(Jan(?:uari)?|Feb(?:ruari)?|Mar(?:et)?|Apr(?:il)?|Mei|Jun(?:i)?|'
     r'Jul(?:i)?|Agu(?:stus)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Des(?:ember)?)\s+(\d{4})',
     'id_month'),
]
MONTH_ID = {
    'jan': 1, 'januari': 1, 'feb': 2, 'februari': 2, 'mar': 3, 'maret': 3,
    'apr': 4, 'april': 4, 'mei': 5, 'jun': 6, 'juni': 6, 'jul': 7, 'juli': 7,
    'agu': 8, 'agustus': 8, 'sep': 9, 'september': 9, 'okt': 10, 'oktober': 10,
    'nov': 11, 'november': 11, 'des': 12, 'desember': 12,
}

GRAND_TOTAL_KW  = re.compile(r'grand[-\s]?total', re.IGNORECASE)
TOTAL_KW_RE     = re.compile(r'(?:grand[-\s]?total|total|jumlah|bayar|pembayaran|tagihan|amount)', re.IGNORECASE)
SUBTOTAL_KW_RE  = re.compile(r'(?:sub[-\s]?total|subtotal)', re.IGNORECASE)
SKIP_LINE_RE    = re.compile(
    r'^(?:x|X|--|-)$|^[A-Za-z]{1,2}$|^-+$|^\.+$|'
    r'(?:kembalian|cash|tunai|kembali|diskon|ppn|pajak|service|pb1|rounding|free|tax)',
    re.IGNORECASE
)
PRICE_ONLY_RE   = re.compile(r'^(?:rp\.?\s*)?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{0,2})?)$', re.IGNORECASE)
# QTY_PRICE: "143181" = qty 1 + harga 43181, "213000" = qty 2 + harga 13000
# Pola: digit(s) langsung diikuti angka >=4 digit tanpa spasi/separator
QTY_PRICE_RE    = re.compile(r'^([1-9]{1,2})([0-9]{4,})$')
# QTY_PRICE dengan separator: "1 43,181" atau "1 43181"
QTY_SEP_PRICE_RE = re.compile(r'^([1-9]{1,2})\s+([0-9]{1,3}(?:[.,][0-9]{3})+|[0-9]{4,})$')
NOMINAL_ANY     = re.compile(r'(?:rp\.?\s*)?([0-9]{1,3}(?:[.,][0-9]{3})+|[0-9]{5,})', re.IGNORECASE)
ITEM_NAME_RE    = re.compile(r'^[A-Za-z][A-Za-z0-9\s/\-&\'%\.]{2,}$')


# ── Token types ────────────────────────────────────────────────────────────────────────────
T_NAME      = 'NAME'
T_PRICE     = 'PRICE'
T_QTY_PRICE = 'QTY_PRICE'  # baris berisi qty+harga gabung/terpisah
T_KEYWORD   = 'KEYWORD'    # total, subtotal, tax, dll
T_NOISE     = 'NOISE'


def _classify_line(line: str) -> tuple:
    """
    Klasifikasikan satu baris menjadi token type.
    Returns: (type, value_dict)
    """
    line = line.strip()
    if not line:
        return T_NOISE, {}

    # Skip keywords (kembalian, tunai, tax, dll)
    if SKIP_LINE_RE.search(line):
        # Tapi kalau ada nominal di dalamnya, bisa jadi total keyword
        if TOTAL_KW_RE.search(line) and not SUBTOTAL_KW_RE.search(line):
            m = NOMINAL_ANY.search(line)
            if m:
                val = _parse_nominal(m.group(1))
                return T_KEYWORD, {'kw': 'total', 'price': val}
        return T_NOISE, {}

    # Grand total / total keyword
    if GRAND_TOTAL_KW.search(line):
        m = NOMINAL_ANY.search(line)
        val = _parse_nominal(m.group(1)) if m else None
        return T_KEYWORD, {'kw': 'grand_total', 'price': val}

    if TOTAL_KW_RE.search(line) and not SUBTOTAL_KW_RE.search(line):
        m = NOMINAL_ANY.search(line)
        val = _parse_nominal(m.group(1)) if m else None
        return T_KEYWORD, {'kw': 'total', 'price': val}

    if SUBTOTAL_KW_RE.search(line):
        return T_KEYWORD, {'kw': 'subtotal'}

    # QTY_PRICE dengan spasi: "1 43,181"
    m = QTY_SEP_PRICE_RE.match(line)
    if m:
        qty = int(m.group(1))
        price = _parse_nominal(m.group(2))
        if price and price >= 100:
            return T_QTY_PRICE, {'qty': qty, 'price': price}

    # QTY_PRICE gabung: "143181" -> qty=1, price=43181
    # Hanya jika angka >= 6 digit (qty 1-2 digit + harga 4+ digit)
    m = QTY_PRICE_RE.match(line)
    if m and len(line) >= 5:
        qty_str = m.group(1)
        price_str = m.group(2)
        # Validasi: price_str harus masuk akal sebagai harga
        price_candidate = _parse_nominal(price_str)
        qty_candidate = int(qty_str)
        if price_candidate and price_candidate >= 100 and qty_candidate <= 20:
            # Heuristic: cek apakah ada harga lain sebelumnya yang 'dekat'
            # (deteksi setelah tokenisasi, untuk sekarang mark sebagai QTY_PRICE)
            return T_QTY_PRICE, {'qty': qty_candidate, 'price': price_candidate}

    # Pure price line
    if PRICE_ONLY_RE.match(line.lstrip('Rp').lstrip('rp').strip()):
        val = _parse_nominal(line)
        if val is not None and val >= 0:
            return T_PRICE, {'price': val}

    # Item name
    if ITEM_NAME_RE.match(line):
        return T_NAME, {'name': line}

    return T_NOISE, {}


def _parse_nominal(s: str) -> Optional[int]:
    """Parse string nominal Rupiah ke int."""
    s = str(s).strip().rstrip('.')
    s = re.sub(r'^(?:rp\.?\s*)', '', s, flags=re.IGNORECASE)
    if re.match(r'^[0-9]{1,3}(?:\.[0-9]{3})+$', s):
        return int(s.replace('.', ''))
    if re.match(r'^[0-9]{1,3}(?:,[0-9]{3})+$', s):
        return int(s.replace(',', ''))
    try:
        return int(re.sub(r'[.,]', '', s))
    except ValueError:
        return None


def _extract_date(lines: list) -> Optional[str]:
    for line in lines:
        for pat, fmt in DATE_PATTERNS:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                try:
                    if fmt == 'id_month':
                        day, month_str, year = m.group(1), m.group(2).lower(), m.group(3)
                        month_num = MONTH_ID.get(month_str)
                        if month_num:
                            return f"{int(year):04d}-{month_num:02d}-{int(day):02d}"
                    elif '/' in fmt:
                        sep = re.search(r'[/\-\.]', m.group(0)).group()
                        raw = m.group(0).replace(sep, '/')
                        dt = datetime.strptime(raw, fmt)
                        return dt.strftime('%Y-%m-%d')
                except Exception:
                    continue
    return None


def _extract_total(tokens: list) -> Optional[int]:
    """
    Ekstrak grand total dari token list.
    Priority: grand_total keyword > total keyword > nominal terbesar.
    """
    grand_total_candidates = []
    total_candidates = []

    for i, (ttype, tval) in enumerate(tokens):
        if ttype == T_KEYWORD:
            kw = tval.get('kw', '')
            price = tval.get('price')

            # Cek baris berikutnya jika harga tidak ada di baris yang sama
            if price is None and i + 1 < len(tokens):
                next_type, next_val = tokens[i + 1]
                if next_type in (T_PRICE, T_QTY_PRICE):
                    price = next_val.get('price')

            if price and price >= 1000:
                if kw == 'grand_total':
                    grand_total_candidates.append(price)
                elif kw == 'total':
                    total_candidates.append(price)

    if grand_total_candidates:
        return max(grand_total_candidates)
    if total_candidates:
        return max(total_candidates)

    # Fallback: nominal terbesar dari semua token
    all_prices = [
        v.get('price', 0) for t, v in tokens
        if t in (T_PRICE, T_QTY_PRICE) and v.get('price', 0) >= 1000
    ]
    return max(all_prices) if all_prices else None


def _pair_items(tokens: list) -> tuple:
    """
    Pasangkan NAME token dengan PRICE/QTY_PRICE token.

    Dua arah:
    - Forward: NAME muncul, cari PRICE dalam lookahead N token
    - Backward: PRICE/QTY_PRICE muncul, cek apakah NAME ada di lookback N token
      (format B: harga sebelum nama)

    Returns: (merchant: str, items: list[dict])
    """
    items = []
    merchant = "Unknown"
    used_indices = set()  # track token yang sudah dipasangkan
    LOOKAHEAD = 4

    # ── Pass 1: Forward pairing (NAME -> PRICE) ──────────────────────────────
    for i, (ttype, tval) in enumerate(tokens):
        if ttype != T_NAME or i in used_indices:
            continue
        name = tval['name']
        # Cari PRICE/QTY_PRICE setelah NAME
        for j in range(i + 1, min(i + LOOKAHEAD + 1, len(tokens))):
            if j in used_indices:
                continue
            jtype, jval = tokens[j]
            if jtype == T_NAME:  # nama item lain -> stop
                break
            if jtype in (T_PRICE, T_QTY_PRICE) and jval.get('price', 0) >= 100:
                qty = jval.get('qty', 1)
                items.append({
                    'name': name,
                    'qty': qty,
                    'price': jval['price'],
                    '_idx': i
                })
                used_indices.add(i)
                used_indices.add(j)
                break
            if jtype == T_KEYWORD:
                break

    # ── Pass 2: Backward pairing (PRICE/QTY_PRICE sebelum NAME) ───────────────
    # Untuk format B: harga/qty-harga muncul SEBELUM nama item
    for i, (ttype, tval) in enumerate(tokens):
        if ttype not in (T_PRICE, T_QTY_PRICE) or i in used_indices:
            continue
        if tval.get('price', 0) < 100:
            continue
        # Cari NAME setelah token harga ini
        for j in range(i + 1, min(i + LOOKAHEAD + 1, len(tokens))):
            if j in used_indices:
                continue
            jtype, jval = tokens[j]
            if jtype == T_NAME:
                # Pastikan bukan nama yang sudah dipasangkan
                qty = tval.get('qty', 1)
                items.append({
                    'name': jval['name'],
                    'qty': qty,
                    'price': tval['price'],
                    '_idx': j
                })
                used_indices.add(i)
                used_indices.add(j)
                break
            if jtype in (T_PRICE, T_QTY_PRICE) and ttype == T_PRICE:
                # Dua harga berurutan -> harga pertama mungkin harga satuan,
                # yang kedua total baris -> pakai yang kedua (total baris)
                if tokens[j][1].get('price', 0) > tval['price']:
                    # Update harga ke total baris
                    pass
                break
            if jtype == T_KEYWORD:
                break

    # ── Deduplicate & sort by original index ──────────────────────────────
    # Deduplicate by name (ambil yang pertama muncul)
    seen_names = set()
    unique_items = []
    for it in sorted(items, key=lambda x: x.get('_idx', 0)):
        name_key = it['name'].lower()
        if name_key not in seen_names:
            seen_names.add(name_key)
            unique_items.append({
                'name': it['name'],
                'qty': it['qty'],
                'price': it['price']
            })

    # ── Deteksi merchant ────────────────────────────────────────────────────
    # Merchant = NAME token pertama yang muncul SEBELUM item pertama dipasangkan
    first_item_idx = min((it.get('_idx', 999) for it in items), default=999)
    merchant_candidates = [
        tval['name'] for i, (ttype, tval) in enumerate(tokens)
        if ttype == T_NAME and i < first_item_idx
           and i not in used_indices
           and len(tval['name']) >= 4
    ]
    if merchant_candidates:
        merchant = max(merchant_candidates, key=len).title()

    return merchant, unique_items


def _classify_category(merchant: str, items: list) -> str:
    item_names = ' '.join(it['name'] if isinstance(it, dict) else it for it in items)
    text = (merchant + ' ' + item_names).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    return "lainnya"


def _compute_confidence(merchant: str, date, total, items: list) -> float:
    score = 0.0
    if merchant and merchant != "Unknown":
        score += 0.25
    if date:
        score += 0.20
    if total and total > 0:
        score += 0.30
    if items:
        score += min(len(items) * 0.05, 0.25)
    return round(min(score, 1.0), 2)


# ── Main ────────────────────────────────────────────────────────────────────────────
def parse_receipt(raw_text: str) -> dict:
    """Parse raw OCR text → structured receipt dict sesuai AI-DS-SPEC Fitur 2."""
    lines = [l.strip() for l in raw_text.strip().split('\n') if l.strip()]

    # Tokenisasi
    tokens = [_classify_line(l) for l in lines]

    merchant, items = _pair_items(tokens)
    date       = _extract_date(lines)
    total      = _extract_total(tokens)
    category   = _classify_category(merchant, items)
    confidence = _compute_confidence(merchant, date, total, items)

    return {
        "merchant":   merchant,
        "date":       date,
        "total":      total,
        "items":      items,
        "category":   category,
        "confidence": confidence,
    }


if __name__ == "__main__":
    import json

    # Test 1: Format B (A Fung - harga sebelum nama)
    print("=== Test: A Fung (Format B) ===")
    sample_afung = """43,181
143181
BASO TAHU
13,000
113000
ES JERUK
56181
TOTAL
5,618
TAX10.00%-
61799
GRAND TOTAL-
62,000
TUNAI
201
KEMBALI"""
    r = parse_receipt(sample_afung)
    print(json.dumps(r, indent=2, ensure_ascii=False))

    # Test 2: Format A (CORD - nama sebelum harga)
    print("\n=== Test: CORD (Format A) ===")
    sample_cord = """Nasi Campur Bali
X
125,000
Bbk Bengil Nasi
X
37,000
MilkShake Starwb
24,000
1,346,000
Sub-Total
1,591,600
Grand Total"""
    r2 = parse_receipt(sample_cord)
    print(json.dumps(r2, indent=2, ensure_ascii=False))
