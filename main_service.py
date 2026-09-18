"""
MAIN SERVICE (OCR + Ceviri servisi)
====================================
Bu servis Playwright ICERMEZ. Sadece:
  - Hesap sistemi (kayit/giris/kitaplik) - sqlite
  - OCR (EasyOCR) + DeepL ceviri + PIL ile Turkce yazi cizme

Sayfa gorsellerini SCRAPER_SERVICE_URL adresindeki ayri servisten,
10'ARLIK GRUPLAR halinde ister. Her grup gelir gelmez hemen islenir
(OCR+ceviri+cizim) ve client'a STREAM edilir (NDJSON), sonra bir
sonraki 10'luk grup istenir. Boylece ne bu serviste ne de scraper
serviste butun bolum ayni anda bellekte durmaz.

Akis (istenen dongu):
    main_service --(url)--> scraper_service /session/start   -> ilk 10 sayfa (URL/base64)
    main_service: bu 10 sayfayi indir+OCR+cevir+ciz -> client'a gonder (stream satiri)
    main_service --(session_id)--> scraper_service /session/next -> sonraki 10 sayfa
    ... has_more=false olana kadar tekrar ...
"""

import os
import io
import re
import json
import base64
import asyncio
import sqlite3
import hashlib
import secrets
import threading
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
from PIL import Image, ImageDraw, ImageFont

# --- AYARLAR ---
# ANAHTAR KODA YAZILMAZ: Render > Environment kismindan DEEPL_API_KEY olarak girilir.
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Kalici disk kullaniyorsan Render'da DB_PATH=/data/app_data.db gibi ayarla.
DB_PATH = os.environ.get("DB_PATH") or os.path.join(BASE_DIR, "app_data.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
EASYOCR_DIR = os.path.join(BASE_DIR, ".easyocr")

# Ayri deploy edilen scraper servisinin adresi (Render'da 2. servis olarak acilacak)
SCRAPER_SERVICE_URL = os.environ.get("SCRAPER_SERVICE_URL", "http://localhost:8001")

BATCH_SIZE = 10
MAX_CONCURRENT_OCR = int(os.environ.get("MAX_CONCURRENT_OCR", "2"))
ocr_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)

app = FastAPI(title="Manga Translator Engine (OCR/Ceviri servisi)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ISTEK MODELLERI ---
class MangaTranslateRequest(BaseModel):
    url: str
    title: Optional[str] = "Untitled"
    layout: Optional[str] = "webtoon"
    source_lang: Optional[str] = "en"
    target_lang: Optional[str] = "tr"


class GetChapterImagesRequest(BaseModel):
    url: str


class TranslateBatchRequest(BaseModel):
    image_urls: List[str]
    source_lang: Optional[str] = "en"
    target_lang: Optional[str] = "tr"


class RegisterRequest(BaseModel):
    username: str
    password: str
    remember_me: Optional[bool] = True


class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: Optional[bool] = True


class LibrarySaveRequest(BaseModel):
    library: List[dict]


# --- HESAP SISTEMI: VERITABANI (degismedi) ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN expires_at TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS libraries (
            user_id INTEGER PRIMARY KEY,
            data TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def hash_password(password: str, salt: Optional[str] = None):
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    test_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(test_hash, stored_hash)


def create_session(conn, user_id: int) -> str:
    token = secrets.token_hex(32)
    conn.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, NULL)",
        (token, user_id),
    )
    return token


def get_current_user(authorization: Optional[str] = Header(None)) -> int:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Yetkilendirme basligi eksik")
    token = authorization.split(" ", 1)[1].strip()
    conn = get_db()
    row = conn.execute(
        "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=401, detail="Gecersiz veya suresi dolmus oturum")

    if row["expires_at"]:
        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
        except ValueError:
            expires_at = None
        if expires_at and datetime.utcnow() > expires_at:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            conn.close()
            raise HTTPException(status_code=401, detail="Oturum suresi doldu, lutfen tekrar giris yapin")

    conn.close()
    return row["user_id"]


# --- HESAP SISTEMI: ENDPOINT'LER (degismedi) ---
@app.post("/api/register")
def register(payload: RegisterRequest):
    username = payload.username.strip()
    password = payload.password

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Kullanici adi en az 3 karakter olmali")
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Sifre en az 4 karakter olmali")

    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Bu kullanici adi zaten alinmis")

    pwd_hash, salt = hash_password(password)
    cursor = conn.execute(
        "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
        (username, pwd_hash, salt),
    )
    user_id = cursor.lastrowid
    conn.execute("INSERT INTO libraries (user_id, data) VALUES (?, ?)", (user_id, "[]"))
    token = create_session(conn, user_id)
    conn.commit()
    conn.close()

    return {"status": "success", "token": token, "username": username}


@app.post("/api/login")
def login(payload: LoginRequest):
    username = payload.username.strip()

    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not verify_password(payload.password, row["salt"], row["password_hash"]):
        conn.close()
        raise HTTPException(status_code=401, detail="Kullanici adi veya sifre hatali")

    token = create_session(conn, row["id"])
    conn.commit()
    conn.close()

    return {"status": "success", "token": token, "username": username}


@app.post("/api/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    return {"status": "success"}


@app.get("/api/me")
def me(user_id: int = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Kullanici bulunamadi")
    return {"status": "success", "username": row["username"]}


@app.get("/api/library")
def get_library(user_id: int = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT data FROM libraries WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    library = json.loads(row["data"]) if row and row["data"] else []
    return {"status": "success", "library": library}


@app.post("/api/library")
def save_library(payload: LibrarySaveRequest, user_id: int = Depends(get_current_user)):
    conn = get_db()
    data_str = json.dumps(payload.library, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO libraries (user_id, data) VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET data = excluded.data
        """,
        (user_id, data_str),
    )
    conn.commit()
    conn.close()
    return {"status": "success"}


# --- OCR LAZY LOADING ---
ocr_reader = None
_OCR_INIT_LOCK = threading.Lock()
OCR_LOCK = threading.Lock()  # readtext ayni anda tek thread'de calissin (RAM + thread guvenligi)


def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        with _OCR_INIT_LOCK:
            if ocr_reader is None:
                import easyocr

                ocr_reader = easyocr.Reader(
                    ["en"],
                    gpu=False,
                    model_storage_directory=EASYOCR_DIR,
                    user_network_directory=EASYOCR_DIR,
                )
    return ocr_reader


def run_ocr(image_bytes: bytes):
    reader = get_ocr_reader()
    with OCR_LOCK:
        return reader.readtext(image_bytes)


# 2. SATIR BIRLESTIRICI (degismedi)
def merge_ocr_lines(results, y_threshold=12):
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r[0][0][1])
    clusters = []

    for bbox, text, prob in sorted_results:
        min_y = min(p[1] for p in bbox)
        max_y = max(p[1] for p in bbox)
        min_x = min(p[0] for p in bbox)
        max_x = max(p[0] for p in bbox)

        merged = False
        for c in clusters:
            c_max_y = max(p[1] for box in c["bboxes"] for p in box)
            c_min_x = min(p[0] for box in c["bboxes"] for p in box)
            c_max_x = max(p[0] for box in c["bboxes"] for p in box)

            if abs(min_y - c_max_y) < y_threshold and not (max_x < c_min_x or min_x > c_max_x):
                c["texts"].append(text)
                c["bboxes"].append(bbox)
                c["probs"].append(prob)
                merged = True
                break

        if not merged:
            clusters.append({"texts": [text], "bboxes": [bbox], "probs": [prob]})

    merged_results = []
    for c in clusters:
        full_text = " ".join(c["texts"])
        all_x = [p[0] for box in c["bboxes"] for p in box]
        all_y = [p[1] for box in c["bboxes"] for p in box]
        merged_bbox = [
            [min(all_x), min(all_y)],
            [max(all_x), min(all_y)],
            [max(all_x), max(all_y)],
            [min(all_x), max(all_y)],
        ]
        avg_prob = sum(c["probs"]) / len(c["probs"])
        merged_results.append((merged_bbox, full_text, avg_prob))

    return merged_results


# 3. SFX VE ANLATICI FILTRESI (degismedi)
def filter_sfx_and_narratives(ocr_results):
    filtered = []
    sfx_words = {
        "vwoom", "wham", "boom", "bam", "thud", "whoosh", "ha", "oh", "uh",
        "aom", "kka", "wow", "gasp", "sigh", "crash", "tap",
    }

    for bbox, text, prob in ocr_results:
        clean_t = text.strip()
        text_lower = clean_t.lower()

        if len(clean_t) <= 1 or text_lower in sfx_words:
            continue

        box_w = max(p[0] for p in bbox) - min(p[0] for p in bbox)
        box_h = max(p[1] for p in bbox) - min(p[1] for p in bbox)

        if box_w < 18 or box_h < 10:
            continue

        if len(clean_t.split()) == 1 and clean_t.isupper() and len(clean_t) <= 4:
            continue

        filtered.append((bbox, clean_t, prob))
    return filtered


# 4. DEEPL XML BAGLAMSAL CEVIRI (degismedi)
def translate_batch_texts(texts: List[str], source_lang: str = "en", target_lang: str = "tr") -> List[str]:
    if not texts:
        return []

    cleaned_texts = [re.sub(r"\s+", " ", t).strip() for t in texts]
    if not any(cleaned_texts):
        return [""] * len(texts)

    if not DEEPL_API_KEY:
        print("[DEEPL] DEEPL_API_KEY tanimli degil, ceviri atlandi")
        return [t.upper() for t in cleaned_texts]

    tagged_input = ""
    for idx, txt in enumerate(cleaned_texts):
        tagged_input += f"<b{idx}>{txt}</b{idx}> "

    try:
        url = "https://api-free.deepl.com/v2/translate"
        headers = {
            "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": [tagged_input.strip()],
            "target_lang": target_lang.upper(),
            "source_lang": source_lang.upper(),
            "tag_handling": "xml",
        }

        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                translated_xml = response.json()["translations"][0]["text"]
                translations = [""] * len(cleaned_texts)

                matches = re.findall(r"<b(\d+)>(.*?)</b\1>", translated_xml, re.DOTALL)
                for idx_str, tr_txt in matches:
                    idx = int(idx_str)
                    if idx < len(translations):
                        translations[idx] = tr_txt.strip().upper()

                for i in range(len(translations)):
                    if not translations[i]:
                        translations[i] = cleaned_texts[i].upper()

                return translations
            else:
                print(f"[DEEPL HATA KODU]: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[DEEPL BAGLANTI HATASI]: {e}")

    return [t.upper() for t in cleaned_texts]


# 5. FONT VE DINAMIK YAZI SIGDIRMA (degismedi)
def get_turkish_font(font_size: int):
    font_paths = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
    ]
    for p in font_paths:
        try:
            return ImageFont.truetype(p, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text_for_box(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines, current_line = [], []

    for word in words:
        test_line = " ".join(current_line + [word])
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
        except Exception:
            w = len(test_line) * (font.size * 0.5 if hasattr(font, "size") else 6)

        if w <= max_width or not current_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))
    return lines


def fit_text_in_box(text: str, max_w: int, max_h: int, draw: ImageDraw.ImageDraw):
    max_w, max_h = max(15, max_w), max(15, max_h)
    font_size = min(max_h, 28)
    min_font_size = 8

    while font_size >= min_font_size:
        font = get_turkish_font(font_size)
        lines = wrap_text_for_box(text, font, max_w, draw)
        line_height = font_size + 2
        total_h = len(lines) * line_height

        if total_h <= max_h or font_size == min_font_size:
            return font, lines, font_size, total_h
        font_size -= 1

    font = get_turkish_font(min_font_size)
    lines = wrap_text_for_box(text, font, max_w, draw)
    return font, lines, min_font_size, len(lines) * (min_font_size + 2)


# 6. ARKA PLAN VE YAZI RENGI TESPITI (degismedi)
def get_dominant_color(image: Image.Image, bbox):
    min_x = max(0, int(min(p[0] for p in bbox)) - 2)
    max_x = min(image.width - 1, int(max(p[0] for p in bbox)) + 2)
    min_y = max(0, int(min(p[1] for p in bbox)) - 2)
    max_y = min(image.height - 1, int(max(p[1] for p in bbox)) + 2)

    pixels = []
    for x in range(min_x, max_x + 1):
        if x < image.width:
            pixels.extend([image.getpixel((x, min_y)), image.getpixel((x, max_y))])
    for y in range(min_y, max_y + 1):
        if y < image.height:
            pixels.extend([image.getpixel((min_x, y)), image.getpixel((max_x, y))])

    if not pixels:
        return (255, 255, 255)

    r = sum(p[0] for p in pixels) // len(pixels)
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    return (r, g, b)


def get_text_color_for_bg(bg_color):
    luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return "white" if luminance < 128 else "black"


# 7. RESME TURKCE METIN CIZME (degismedi)
def process_and_draw_translation(image: Image.Image, ocr_results, translated_texts: List[str]) -> Image.Image:
    draw = ImageDraw.Draw(image)

    for (bbox, text, prob), tr_text in zip(ocr_results, translated_texts):
        tr_text = tr_text.strip()
        if not tr_text:
            continue

        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        min_x, max_x = int(min(x_coords)), int(max(x_coords))
        min_y, max_y = int(min(y_coords)), int(max(y_coords))

        box_w, box_h = max(15, max_x - min_x), max(12, max_y - min_y)

        bg_color = get_dominant_color(image, bbox)
        text_color = get_text_color_for_bg(bg_color)

        draw.rectangle([min_x - 2, min_y - 2, max_x + 2, max_y + 2], fill=bg_color)
        font, lines, font_size, total_h = fit_text_in_box(tr_text, box_w, box_h, draw)

        y_start = min_y + (box_h - total_h) / 2
        line_height = font_size + 2

        for idx, line in enumerate(lines):
            try:
                line_w = draw.textbbox((0, 0), line, font=font)[2]
            except Exception:
                line_w = len(line) * (font_size * 0.5)

            x_start = min_x + (box_w - line_w) / 2
            current_y = y_start + (idx * line_height)
            draw.text((x_start, current_y), line, fill=text_color, font=font)

    return image


def _translate_and_render(image, merged_valid_results, raw_texts, source_lang, target_lang):
    """Senkron/agir isler: DeepL + PIL cizimi + JPEG kodlama. Event loop'u bloklamamak icin thread'de calisir."""
    translated_texts = translate_batch_texts(raw_texts, source_lang, target_lang)
    processed_image = process_and_draw_translation(image, merged_valid_results, translated_texts)
    buffered = io.BytesIO()
    processed_image.save(buffered, format="JPEG", quality=90)
    b64_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return translated_texts, f"data:image/jpeg;base64,{b64_encoded}"


# 8. TEK BIR RESMI CEVIRME
async def process_single_image(client: httpx.AsyncClient, img_url: str, source_lang: str, target_lang: str):
    bubbles = []
    base64_image_url = ""
    try:
        if img_url.startswith("data:image"):
            header, encoded = img_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        else:
            img_resp = await client.get(
                img_url,
                timeout=20.0,
                headers={"Referer": img_url, "User-Agent": "Mozilla/5.0"},
            )
            if img_resp.status_code == 200:
                image_bytes = img_resp.content
            else:
                return img_url, []

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = image.size

        results = await asyncio.to_thread(run_ocr, image_bytes)

        valid_results = [
            r for r in results if r[2] > 0.30 and len(re.sub(r"[^a-zA-Z]", "", r[1])) >= 2
        ]

        filtered_results = filter_sfx_and_narratives(valid_results)
        merged_valid_results = merge_ocr_lines(filtered_results)
        raw_texts = [r[1] for r in merged_valid_results]

        translated_texts, base64_image_url = await asyncio.to_thread(
            _translate_and_render, image, merged_valid_results, raw_texts, source_lang, target_lang
        )

        for b_id, ((bbox, orig_text, _), tr_text) in enumerate(
            zip(merged_valid_results, translated_texts), start=1
        ):
            x_coord = float(bbox[0][0]) / img_w if img_w > 0 else 0.0
            y_coord = float(bbox[0][1]) / img_h if img_h > 0 else 0.0

            bubbles.append(
                {
                    "id": b_id,
                    "original_text": orig_text,
                    "translated_text": tr_text,
                    "x": round(x_coord, 4),
                    "y": round(y_coord, 4),
                }
            )
        # Buyuk objeleri elden birakalim (GC'ye yardim)
        del image, image_bytes
    except Exception as ocr_error:
        print(f"Gorsel Isleme Hatasi: {ocr_error}")

    final_url = base64_image_url if base64_image_url else img_url
    return final_url, bubbles


async def bounded_process_single_image(client, img_url, s_lang, t_lang):
    async with ocr_semaphore:
        return await process_single_image(client, img_url, s_lang, t_lang)


async def process_batch(image_urls: List[str], source_lang: str, target_lang: str, page_offset: int) -> List[dict]:
    """Bir 10'luk grubu indirir+OCR+cevirir+cizer, sonuclari doner. Islem bitince bellek serbest kalir."""
    async with httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
    ) as client:
        tasks = [bounded_process_single_image(client, u, source_lang, target_lang) for u in image_urls]
        results = await asyncio.gather(*tasks)

    pages = []
    for i, (img_url, (img_result_url, bubbles)) in enumerate(zip(image_urls, results)):
        pages.append(
            {
                "page_number": page_offset + i + 1,
                "image_url": img_result_url,
                "original_url": img_url if not img_url.startswith("data:image") else "CANVAS_BASE64",
                "bubbles": bubbles,
            }
        )
    return pages


# --- SCRAPER SERVISI ILE KONUSMA (10'arli dongu) ---
async def _scraper_stream_batches(url: str, http_client: httpx.AsyncClient):
    """scraper_service'den 10'ar 10'ar sayfa gruplarini verir. Bitince/hata olunca oturumu kapatir."""
    resp = await http_client.post(f"{SCRAPER_SERVICE_URL}/session/start", json={"url": url}, timeout=120.0)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Scraper servisi hata verdi ({resp.status_code}): {resp.text}")
    data = resp.json()
    session_id = data.get("session_id")

    try:
        yield data["images"]
        has_more = data.get("has_more", False)

        while session_id and has_more:
            resp = await http_client.post(
                f"{SCRAPER_SERVICE_URL}/session/next", json={"session_id": session_id}, timeout=120.0
            )
            if resp.status_code != 200:
                # Sessizce birakma: kullanici eksik bolumu basarili sanmasin
                raise HTTPException(
                    status_code=502, detail=f"Scraper /session/next hata verdi ({resp.status_code}): {resp.text}"
                )
            data = resp.json()
            yield data["images"]
            has_more = data.get("has_more", False)
    finally:
        # Kullanici vazgecse / hata olsa da scraper'daki Chromium oturumu kapansin
        if session_id:
            try:
                await http_client.post(
                    f"{SCRAPER_SERVICE_URL}/session/close", json={"session_id": session_id}, timeout=10.0
                )
            except Exception:
                pass  # scraper'daki TTL temizligi yedek olarak zaten var


@app.on_event("startup")
async def _preload_ocr():
    """OCR modelini arka planda yukle: ilk istek beklemesin, port hemen acilsin."""
    if os.environ.get("PRELOAD_OCR", "1") == "1":
        asyncio.create_task(asyncio.to_thread(get_ocr_reader))


@app.get("/")
def root():
    return {"status": "ok", "message": "Manga Translator Engine (OCR/Ceviri) Aktif"}


@app.post("/api/get-chapter-images")
async def get_chapter_images(payload: GetChapterImagesRequest, user_id: int = Depends(get_current_user)):
    """Geriye donuk uyumluluk icin: tum sayfalari toplayip tek seferde doner (scraper servisine proxy)."""
    if not payload.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Gecersiz URL formati")

    all_images: List[str] = []
    async with httpx.AsyncClient() as http_client:
        async for batch in _scraper_stream_batches(payload.url, http_client):
            all_images.extend(batch)

    if not all_images:
        raise HTTPException(status_code=404, detail="Bolum ici gorseller cekilemedi.")

    return {"status": "success", "total_pages": len(all_images), "images": all_images}


@app.post("/api/translate-page-batch")
async def translate_page_batch(payload: TranslateBatchRequest, user_id: int = Depends(get_current_user)):
    """Client'in kendi elindeki bir URL grubunu (ornegin 10'luk) direkt cevirmesi icin."""
    if not payload.image_urls:
        return {"status": "success", "pages": []}

    pages = await process_batch(payload.image_urls, payload.source_lang or "en", payload.target_lang or "tr", 0)
    return {"status": "success", "pages": pages}


@app.post("/api/translate-chapter")
async def translate_chapter(payload: MangaTranslateRequest, user_id: int = Depends(get_current_user)):
    """
    STREAMING endpoint: scraper servisinden 10'ar 10'ar sayfa ister, her grubu
    isler ve HEMEN client'a bir NDJSON satiri olarak gonderir. Boylece:
      - Ne scraper ne bu servis butun bolumu ayni anda bellekte tutmaz.
      - Client (Flutter) sayfalari geldikce ekrana ekleyebilir.

    Yanit formati: her satir bagimsiz bir JSON objesi (\\n ile ayrilir).
        {"type": "meta", "title": ..., "layout": ..., ...}
        {"type": "pages", "pages": [ {...sayfa...}, ... ]}   <- her 10'luk grup icin bir tane
        {"type": "done", "total_pages": N}
        {"type": "error", "detail": "..."}
    """
    if not payload.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Gecersiz URL formati")

    source_lang = payload.source_lang or "en"
    target_lang = payload.target_lang or "tr"

    async def stream():
        total = 0
        meta = {
            "type": "meta",
            "title": payload.title or "Isimsiz Manga",
            "layout": payload.layout,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        yield json.dumps(meta, ensure_ascii=False) + "\n"

        try:
            async with httpx.AsyncClient() as http_client:
                async for image_batch in _scraper_stream_batches(payload.url, http_client):
                    if not image_batch:
                        continue
                    pages = await process_batch(image_batch, source_lang, target_lang, total)
                    total += len(pages)
                    yield json.dumps({"type": "pages", "pages": pages}, ensure_ascii=False) + "\n"
                    # bir sonraki 10'luk gruba gecmeden once bu grubun buyuk
                    # nesnelerini (pages listesi disinda tuttugumuz yok) serbest birak
        except HTTPException as e:
            yield json.dumps({"type": "error", "detail": e.detail}, ensure_ascii=False) + "\n"
            return
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False) + "\n"
            return

        if total == 0:
            yield json.dumps({"type": "error", "detail": "Bolum ici gorseller cekilemedi."}, ensure_ascii=False) + "\n"
        else:
            yield json.dumps({"type": "done", "total_pages": total}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
