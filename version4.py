# coastal_safety_app.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tempfile
import cv2
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import zipfile
import math
from collections import deque
import matplotlib.pyplot as plt
from fpdf import FPDF

# ---------- NEW imports for messaging + TTS ----------
import requests
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email import encoders
from gtts import gTTS

# Twilio import may not be installed — guard usage
try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None

# ---------- Multilingual strings ----------
# Basic translations for key UI text. You can expand these JSONs as needed.
LANGS = {
    "en": {
        "header": "🌊 Coastal Safety AI — Rip Current & Human Risk Detection",
        "desc": "Uses YOLO COCO classes (person, boat, surfboard) + heuristics for rip currents, crowding, rock proximity, and possible drowning risk.",
        "controls": "Controls & Parameters",
        "start": "▶️ Start",
        "stop": "⏹️ Stop",
        "alert_sound": "Alert sound (optional):",
        "download_pdf": "Download PDF Analysis",
        "download_zip": "Download snapshots ZIP",
        "export_csv": "Export log CSV",
        "session": "Session",
        "last_snapshot": "Last alert snapshot",
        "no_alerts": "No alerts recorded yet.",
    },
    "hi": {
        "header": "🌊 तट सुरक्षा AI — रिप करंट और मानवीय जोखिम पहचान",
        "desc": "YOLO COCO क्लासेस (person, boat, surfboard) और नियम-आधारित चेतावनियाँ।",
        "controls": "नियंत्रण और पैरामीटर",
        "start": "▶️ प्रारंभ",
        "stop": "⏹️ रोकें",
        "alert_sound": "अलर्ट ध्वनि (वैकल्पिक):",
        "download_pdf": "PDF विश्लेषण डाउनलोड करें",
        "download_zip": "स्नैपशॉट्स ZIP डाउनलोड करें",
        "export_csv": "लॉग CSV एक्सपोर्ट करें",
        "session": "सत्र",
        "last_snapshot": "अंतिम अलर्ट स्नैपशॉट",
        "no_alerts": "अभी कोई अलर्ट रिकॉर्ड नहीं है।"
    },
    "te": {
        "header": "🌊 కోస్టల్ సేఫ్టీ AI — రిప్ కరంట్ & మానవ ప్రమాద గుర్తింపు",
        "desc": "YOLO COCO తరగతులు (person, boat, surfboard) + హెయూరిస్టిక్స్.",
        "controls": " నియంత్రణలు & పరామితులు",
        "start": "▶️ ప్రారంభం",
        "stop": "⏹️ ఆపు",
        "alert_sound": "సత్వర హెచ్చరిక ధ్వని (ঐচ্ছిక):",
        "download_pdf": "PDF విశ్లేషణ డౌన్లోడ్",
        "download_zip": "స్పాషాట్స్ ZIP డౌన్లోడ్",
        "export_csv": "లాగ్ CSV ఎక్స్‌పోర్ట్",
        "session": "సెషన్",
        "last_snapshot": "చివరి అలర్ట్ స్నాప్‌షాట్",
        "no_alerts": "ఇంకా అలర్ట్స్ నమోదు కాలేదు."
    },
    # Add more languages here (ta, kn, ml, mr, gu, bn, or, pa...)
}

# ---------- Helper: use selected language ----------
def L(key):
    lang = st.session_state.get("ui_lang", "en")
    return LANGS.get(lang, LANGS["en"]).get(key, LANGS["en"].get(key, key))

# === Sound Alert Function (autoplay HTML) ===
def play_sound_auto_bytes(mp3_bytes):
    """Play mp3 bytes via Streamlit HTML audio tag (autoplay)."""
    try:
        b64 = base64.b64encode(mp3_bytes).decode()
        audio_tag = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_tag, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"Failed to play sound: {e}")

# Keep old function name for compatibility (file path)
def play_sound_auto(sound_path):
    try:
        with open(sound_path, "rb") as f:
            data = f.read()
        play_sound_auto_bytes(data)
    except Exception as e:
        st.sidebar.warning(f"Failed to play sound: {e}")

# === Colors / Theme ===
PRIMARY_COLOR = "#0ea5e9"
ERROR_COLOR = "#ef4444"
SUCCESS_COLOR = "#10b981"
BG_COLOR = "#0b1220"
TEXT_COLOR = "#e6eef6"

st.set_page_config(layout="wide", page_title="Coastal Safety AI", page_icon="🌊")

st.markdown(f"""
<style>
    .main-header {{ font-size: 2.1rem; color: {PRIMARY_COLOR}; text-align: center; margin-bottom: 0.5rem; }}
    .info-box {{ background-color: #ffffff; padding: 14px; border-radius: 8px; margin-bottom: 12px; color: #0b1220; }}
    .status-ok {{ color: {SUCCESS_COLOR}; font-weight: bold; }}
    .violation-alert {{ color: {ERROR_COLOR}; font-weight: bold; }}
    .stApp {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
    h1, h2, h3 {{ color: {PRIMARY_COLOR}; }}
    .stButton>button {{ background-color: {PRIMARY_COLOR}; color: white; border-radius: 6px; }}
</style>
""", unsafe_allow_html=True)

# ---------- UI LANGUAGE SELECTION ----------
if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "en"

with st.sidebar:
    st.markdown("<h4 style='color: #0ea5e9'>Language / भाषा</h4>", unsafe_allow_html=True)
    lang_choice = st.selectbox("Select language / भाषा चुनें", options=list(LANGS.keys()), index=list(LANGS.keys()).index(st.session_state["ui_lang"]))
    st.session_state["ui_lang"] = lang_choice

st.markdown(f'<h1 class="main-header">{L("header")}</h1>', unsafe_allow_html=True)
st.markdown(L("desc"))

# === Session State Initialization ===
for key, default in {
    "log": [], "run": False, "alert_count": 0, "last_snapshot": None,
    "start_time": None, "total_session_time": 0, "violation_images": [],
    "tracks": {}, "next_track_id": 0, "track_history": {}, "fps_history": [],
    "last_sound_time": 0.0, "alert_map": None, "alert_flag": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# === Sidebar Controls ===
with st.sidebar:
    st.markdown(f"<h3 style='color: #0ea5e9'>{L('controls')}</h3>", unsafe_allow_html=True)
    model_choice = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    confidence_threshold = st.slider("Detection confidence", 0.1, 1.0, 0.4, 0.05)
    run_source = st.radio("Input source", ["Webcam (camera 0)", "Upload video file"], index=0)
    uploaded_file = None
    if run_source == "Upload video file":
        uploaded_file = st.file_uploader("Choose video", type=["mp4", "mov", "avi", "mkv"])
    auto_snapshot = st.checkbox("Auto-capture violation snapshots", value=True)
    show_debug = st.checkbox("Show debug overlays / stats", value=False)

    st.markdown("---")
    st.markdown("<b>Zone & detection parameters</b>", unsafe_allow_html=True)
    safe_line_frac = st.slider("Safe-line (shoreline) - fraction from top (0) to bottom (1)", 0.25, 0.85, 0.55, 0.01)
    deep_water_frac = st.slider("Deep water threshold (fraction from top)", 0.6, 0.98, 0.78, 0.01)
    crowd_threshold = st.slider("Crowd count threshold (wave-impact zone)", 1, 20, 6)
    rip_min_people = st.slider("Rip-current min people in channel", 1, 8, 2)
    rip_speed_thresh = st.slider("Rip-current seaward speed px/s (approx)", 20, 400, 120)
    stationary_time_thresh = st.slider("Stationary time for drowning alert (s)", 6, 60, 12)
    proximity_rock_px = st.slider("Rock proximity threshold (px)", 20, 200, 80)

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button(L("start"), use_container_width=True):
        st.session_state.run = True
        st.session_state.start_time = time.time()
    if col2.button(L("stop"), use_container_width=True):
        if st.session_state.run and st.session_state.start_time:
            st.session_state.total_session_time += time.time() - st.session_state.start_time
        st.session_state.run = False

    # sounds
    st.markdown("---")
    st.markdown(L("alert_sound"))
    sound_path = st.text_input("Path to alert mp3 (leave blank to skip sound)", value="")
    sound_cooldown = st.slider("Sound cooldown (seconds) - prevents alert flood", 0.1, 5.0, 1.0, 0.1)

    # ---------- NEW: Messaging & TTS config ----------
    st.markdown("---")
    st.markdown("<b>Alert Dispatch Configuration</b>", unsafe_allow_html=True)
    st.markdown("Twilio (SMS / WhatsApp) credentials")
    twilio_sid = st.text_input("Twilio Account SID", value="", key="tw_sid")
    twilio_token = st.text_input("Twilio Auth Token", value="", key="tw_token")
    twilio_from = st.text_input("Twilio From Number (E.164) or WhatsApp sender", value="", key="tw_from")
    st.markdown("Telegram Bot")
    telegram_bot_token = st.text_input("Telegram Bot Token", value="", key="tg_token")
    telegram_chat_id = st.text_input("Telegram Chat ID", value="", key="tg_chat")
    st.markdown("Email (SMTP)")
    smtp_email = st.text_input("Sender Email (SMTP login)", value="", key="smtp_email")
    smtp_password = st.text_input("SMTP Password / App Password", value="", type="password", key="smtp_pass")
    smtp_server = st.text_input("SMTP Server (host:port)", value="smtp.gmail.com:587", key="smtp_srv")
    st.markdown("Recipients (comma-separated)")
    recipient_sms = st.text_input("Emergency phone numbers (comma-separated, E.164)", value="", key="rec_sms")
    recipient_whatsapp = st.text_input("WhatsApp numbers (comma-separated, E.164)", value="", key="rec_wa")
    recipient_emails = st.text_input("Recipient emails (comma-separated)", value="", key="rec_em")
    recipient_telegram = st.text_input("Recipient Telegram chat IDs (comma-separated)", value="", key="rec_tg")

    st.markdown("---")
    st.markdown("<b>Georeference (optional)</b>", unsafe_allow_html=True)
    st.markdown("If you provide two known points, pixel coordinates can be mapped to approx GPS.")
    top_left_lat = st.text_input("Top-left GPS: lat", value="", key="geo_tl_lat")
    top_left_lon = st.text_input("Top-left GPS: lon", value="", key="geo_tl_lon")
    bottom_right_lat = st.text_input("Bottom-right GPS: lat", value="", key="geo_br_lat")
    bottom_right_lon = st.text_input("Bottom-right GPS: lon", value="", key="geo_br_lon")

    st.markdown("---")
    st.markdown("<b>Voice TTS</b>", unsafe_allow_html=True)
    tts_provider = st.selectbox("TTS Provider (gTTS free / ElevenLabs placeholder / Google placeholder)", ["gTTS", "ElevenLabs (API)", "Google TTS (API)"])
    default_tts_lang = st.selectbox("TTS language (gTTS codes)", ["en", "hi", "te", "ta", "kn", "ml", "mr", "gu", "bn", "or", "pa"], index=0)
    eleven_api_key = st.text_input("ElevenLabs API key (optional)", value="", key="eleven_key")
    google_tts_key = st.text_input("Google TTS Key (optional)", value="", key="google_key")

    st.markdown("---")
    st.markdown("<b>Optional: Public base URL to serve snapshots (for WhatsApp/SMS media links)</b>", unsafe_allow_html=True)
    public_media_base = st.text_input("Public media base URL (e.g., https://mybucket.s3.amazonaws.com/alerts/). Leave blank to attach via email only.", value="")

# === Sidebar: Predefined rock polygons and wave-impact zone ===
st.sidebar.markdown("---")
st.sidebar.markdown("Define static areas (you can tune these visually while running):")
st.sidebar.markdown("- Rocks: list of polygons (x1,y1;x2,y2;...) in normalized coords (0-1). Example: `0.1,0.7;0.2,0.68;0.18,0.75`")
st.sidebar.markdown("- Wave-impact zone: polygon or rectangle near shore where waves break")
rock_polys_text = st.sidebar.text_area("Rocks polygons (one per line) - normalized coords", value="0.05,0.75;0.12,0.73;0.15,0.80")
wave_zone_text = st.sidebar.text_area("Wave-impact polygon (single) - normalized coords", value="0.15,0.55;0.85,0.55;0.85,0.70;0.15,0.70")

def parse_norm_poly(text):
    try:
        pts = []
        for part in text.split(";"):
            x,y = part.split(",")
            pts.append((float(x.strip()), float(y.strip())))
        return pts
    except:
        return []

rock_polys = [parse_norm_poly(line) for line in rock_polys_text.splitlines() if line.strip()]
wave_zone = parse_norm_poly(wave_zone_text)

# === Utility functions ===
def norm_to_px(poly, w, h):
    return [(int(x*w), int(y*h)) for x,y in poly]

def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        if ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9) + x1):
            inside = not inside
    return inside

def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ---------- NEW: Georeference pixel->latlon mapping ----------
def pixel_to_latlon(px, py, frame_w, frame_h):
    """Map pixel coordinate px,py (int) to lat/lon using user-provided top-left and bottom-right GPS.
       Returns (lat, lon) as floats or None if georef not configured."""
    try:
        if (not st.session_state.get("geo_tl_lat")) or (not st.session_state.get("geo_br_lat")):
            # Use values from sidebar inputs if present
            tl_lat = float(st.session_state.get("geo_tl_lat") or top_left_lat or 0)
            tl_lon = float(st.session_state.get("geo_tl_lon") or top_left_lon or 0)
            br_lat = float(st.session_state.get("geo_br_lat") or bottom_right_lat or 0)
            br_lon = float(st.session_state.get("geo_br_lon") or bottom_right_lon or 0)
            # store for quick access
            st.session_state["geo_tl_lat"] = tl_lat
            st.session_state["geo_tl_lon"] = tl_lon
            st.session_state["geo_br_lat"] = br_lat
            st.session_state["geo_br_lon"] = br_lon
        else:
            tl_lat = float(st.session_state["geo_tl_lat"])
            tl_lon = float(st.session_state["geo_tl_lon"])
            br_lat = float(st.session_state["geo_br_lat"])
            br_lon = float(st.session_state["geo_br_lon"])
    except Exception:
        return None

    # linear interpolation
    lat = tl_lat + (py / float(frame_h)) * (br_lat - tl_lat)
    lon = tl_lon + (px / float(frame_w)) * (br_lon - tl_lon)
    return (round(lat, 6), round(lon, 6))

# === Simple centroid tracker ===
MAX_LOST_FRAMES = 8
def update_tracks(detections_centroids, frame_time):
    tracks = st.session_state["tracks"]
    assigned = set()
    for cid, c in enumerate(detections_centroids):
        best_id = None
        best_dist = 1e9
        for tid, t in tracks.items():
            if t["lost"] > 0:
                continue
            last_cent = t["centroid"]
            d = euclid(c, last_cent)
            if d < best_dist and d < 100:
                best_dist = d
                best_id = tid
        if best_id is not None:
            tracks[best_id]["centroid"] = c
            tracks[best_id]["last_seen"] = frame_time
            tracks[best_id]["lost"] = 0
            tracks[best_id]["trace"].append((frame_time, c))
            assigned.add(cid)
        else:
            tid = st.session_state["next_track_id"]
            st.session_state["next_track_id"] += 1
            tracks[tid] = {"centroid": c, "first_seen": frame_time, "last_seen": frame_time, "lost":0, "trace":deque([(frame_time, c)], maxlen=50)}
            assigned.add(cid)

    for tid, t in list(tracks.items()):
        if abs(t["last_seen"] - frame_time) > 0.001:
            t["lost"] += 1
        if t["lost"] > MAX_LOST_FRAMES:
            if tid not in st.session_state["track_history"]:
                st.session_state["track_history"][tid] = t
            del tracks[tid]

def track_velocity(track):
    trace = track["trace"]
    if len(trace) < 2:
        return (0.0, 0.0)
    (t0, p0), (t1, p1) = trace[-2], trace[-1]
    dt = t1 - t0 if (t1 - t0) > 1e-6 else 1e-6
    vx = (p1[0]-p0[0]) / dt
    vy = (p1[1]-p0[1]) / dt
    return (vx, vy)

# === Model load ===
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(model_choice)
    st.sidebar.success("Model loaded")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# ---------- Messaging functions ----------
def safe_split_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]

def send_sms_via_twilio(message, to_numbers):
    if not twilio_sid or not twilio_token or not twilio_from:
        st.sidebar.warning("Twilio not configured; cannot send SMS/WhatsApp.")
        return False
    if TwilioClient is None:
        st.sidebar.warning("Twilio library not installed.")
        return False
    try:
        client = TwilioClient(twilio_sid, twilio_token)
        for num in to_numbers:
            # standard SMS
            client.messages.create(body=message, from_=twilio_from, to=num)
        return True
    except Exception as e:
        st.sidebar.error(f"Twilio SMS error: {e}")
        return False

def send_whatsapp_via_twilio(message, to_numbers, media_url=None):
    if not twilio_sid or not twilio_token or not twilio_from:
        st.sidebar.warning("Twilio not configured; cannot send WhatsApp.")
        return False
    if TwilioClient is None:
        st.sidebar.warning("Twilio library not installed.")
        return False
    try:
        client = TwilioClient(twilio_sid, twilio_token)
        for num in to_numbers:
            kwargs = {"body": message, "from_": twilio_from, "to": f"whatsapp:{num}"}
            if media_url:
                kwargs["media_url"] = [media_url]
            client.messages.create(**kwargs)
        return True
    except Exception as e:
        st.sidebar.error(f"Twilio WhatsApp error: {e}")
        return False

def send_telegram(message, chat_ids):
    if not telegram_bot_token:
        st.sidebar.warning("Telegram bot token not configured.")
        return False
    try:
        for cid in chat_ids:
            url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
            r = requests.post(url, json={"chat_id": cid, "text": message})
            # ignore response for now
        return True
    except Exception as e:
        st.sidebar.error(f"Telegram error: {e}")
        return False

def send_email_with_attachment(subject, body, to_emails, attachment_bytes=None, attachment_name="snapshot.jpg"):
    if not smtp_email or not smtp_password or not smtp_server:
        st.sidebar.warning("SMTP not configured.")
        return False
    try:
        host, port = smtp_server.split(":")
        port = int(port)
    except Exception:
        st.sidebar.error("SMTP server must be in host:port format (e.g., smtp.gmail.com:587).")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_email
        msg["To"] = ",".join(to_emails)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        if attachment_bytes is not None:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
            msg.attach(part)
        server = smtplib.SMTP(host, port)
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, to_emails, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.sidebar.error(f"SMTP error: {e}")
        return False

# ---------- TTS / Voice generation ----------
def generate_voice_mp3_bytes(text, lang=default_tts_lang):
    """Generate mp3 bytes using gTTS or other providers (placeholders for ElevenLabs/Google)."""
    provider = tts_provider
    if provider == "gTTS":
        try:
            tts = gTTS(text=text, lang=lang)
            f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tts.save(f.name)
            with open(f.name, "rb") as ff:
                data = ff.read()
            try:
                os.unlink(f.name)
            except:
                pass
            return data
        except Exception as e:
            st.sidebar.error(f"gTTS error: {e}")
            return None
    elif provider == "ElevenLabs (API)":
        # Placeholder: user must provide API key in sidebar (eleven_api_key)
        if not eleven_api_key:
            st.sidebar.warning("ElevenLabs key not provided.")
            return None
        # Simple example—user must adapt to ElevenLabs docs
        try:
            url = "https://api.elevenlabs.io/v1/text-to-speech"
            headers = {"xi-api-key": eleven_api_key}
            payload = {"text": text, "voice": "alloy"}  # check ElevenLabs API
            r = requests.post(url, json=payload, headers=headers)
            if r.status_code == 200:
                return r.content
            else:
                st.sidebar.error(f"ElevenLabs error: {r.status_code} {r.text}")
                return None
        except Exception as e:
            st.sidebar.error(f"ElevenLabs request error: {e}")
            return None
    elif provider == "Google TTS (API)":
        # Placeholder: user must provide Google credentials & adapt
        if not google_tts_key:
            st.sidebar.warning("Google TTS key not provided.")
            return None
        st.sidebar.warning("Google TTS support in this example is a placeholder; integrate using Google Cloud client libraries.")
        return None
    else:
        st.sidebar.warning("Unknown TTS provider selected.")
        return None

# ---------- Alert dispatcher ----------
def dispatch_alerts(message_text, frame_bgr=None, approx_coords=None):
    """Send multi-channel alert. frame_bgr is a cv2 BGR image (numpy) or None."""
    # Prepare recipients
    sms_list = safe_split_csv(recipient_sms or "")
    wa_list = safe_split_csv(recipient_whatsapp or "")
    email_list = safe_split_csv(recipient_emails or "")
    tg_list = safe_split_csv(recipient_telegram or "") or safe_split_csv(telegram_chat_id or "")

    # Convert frame to JPEG bytes (for email attachment and optional public upload)
    media_url = None
    img_bytes = None
    if frame_bgr is not None:
        try:
            _, buf = cv2.imencode(".jpg", frame_bgr)
            img_bytes = buf.tobytes()
            # If public_media_base is set, upload is required (not implemented here).
            # We include a helper notice: if user uploads snapshot elsewhere and provides URL, that URL will be sent.
            # Otherwise, we attach image via email.
            if public_media_base:
                # Construct a filename and tell user they should upload file to public_media_base via their infra or S3.
                fn = f"alert_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                media_url = public_media_base.rstrip("/") + "/" + fn
                # Note: actual upload is not implemented — user must upload img_bytes to that URL or S3 bucket.
        except Exception as e:
            st.sidebar.warning(f"Failed to encode snapshot image: {e}")
            img_bytes = None

    # Append coordinates to message if available
    if approx_coords:
        message_text = f"{message_text}\nCoordinates: {approx_coords[0]}°, {approx_coords[1]}°"

    # Send SMS (plain text)
    if sms_list:
        try:
            send_sms_via_twilio(message_text, sms_list)
        except Exception as e:
            st.sidebar.error(f"SMS dispatch error: {e}")

    # Send WhatsApp (with media_url if available)
    if wa_list:
        try:
            send_whatsapp_via_twilio(message_text, wa_list, media_url=media_url)
        except Exception as e:
            st.sidebar.error(f"WhatsApp dispatch error: {e}")

    # Send Telegram
    if tg_list:
        try:
            send_telegram(message_text, tg_list)
        except Exception as e:
            st.sidebar.error(f"Telegram dispatch error: {e}")

    # Send Email (attach snapshot if available)
    if email_list:
        subject = "Coastal Safety AI - URGENT ALERT"
        body = message_text + "\n\nThis alert was generated by Coastal Safety AI."
        try:
            send_email_with_attachment(subject, body, email_list, attachment_bytes=img_bytes, attachment_name="alert_snapshot.jpg")
        except Exception as e:
            st.sidebar.error(f"Email dispatch error: {e}")

    # Generate voice alert & play it
    try:
        # Choose language from default_tts_lang
        voice_bytes = generate_voice_mp3_bytes(message_text, lang=default_tts_lang)
        if voice_bytes:
            play_sound_auto_bytes(voice_bytes)
    except Exception as e:
        st.sidebar.error(f"Voice alert error: {e}")

# === Main layout ===
col_main, col_side = st.columns([3,1])
with col_side:
    total_time = st.session_state.total_session_time + (time.time() - st.session_state.start_time if st.session_state.run and st.session_state.start_time else 0)
    h = int(total_time // 3600)
    m = int((total_time % 3600) // 60)
    s = int(total_time % 60)
    st.markdown(f"<div class='info-box'><h4>{L('session')}</h4><p><b>Time:</b> {h:02}:{m:02}:{s:02}</p><p><b>Alerts:</b> {st.session_state.alert_count}</p></div>", unsafe_allow_html=True)

    if st.session_state.last_snapshot is not None:
        st.markdown(f"<div class='info-box'><h4>{L('last_snapshot')}</h4></div>", unsafe_allow_html=True)
        st.image(st.session_state.last_snapshot, channels="BGR")

    # Export CSV
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(L("export_csv"), csv, f"coastal_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    # Download snapshots zip
    if st.session_state.violation_images:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for i, item in enumerate(st.session_state.violation_images):
                image_data = base64.b64decode(item["image"])
                filename = f'{item["time"].replace(":", "-").replace(" ", "_")}_{item["reason"].replace(" ", "_")}.jpg'
                zip_file.writestr(filename, image_data)
        zip_buffer.seek(0)
        st.download_button(L("download_zip"), zip_buffer, file_name=f"coastal_snapshots_{datetime.datetime.now().strftime('%Y%m%d')}.zip", mime="application/zip")

    # === PDF report generation (textual with heatmaps) ===
    def generate_pdf_report():
        if not st.session_state.violation_images:
            st.warning(L("no_alerts"))
            return

        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Coastal Safety AI - Alert Report", ln=True, align='C')
        pdf.ln(5)
    
        # --- Summary Statistics ---
        pdf.set_font("Arial", '', 12)
        total_alerts = len(st.session_state.violation_images)
        pdf.cell(0, 8, f"Total Alerts Recorded: {total_alerts}", ln=True)
        
        # Convert log to DataFrame
        df_log = pd.DataFrame(st.session_state.log)
        if not df_log.empty:
            alert_types = df_log['Alert'].value_counts()
            pdf.cell(0, 8, f"Unique Alert Types: {len(alert_types)}", ln=True)
            pdf.ln(3)

        # --- Alerts Table ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Alerts Detail:", ln=True)
        pdf.set_font("Arial", '', 10)
        for i, row in df_log.iterrows():
            pdf.multi_cell(0, 6, f"{row['Time']} - {row['Alert']} - {row['Details']}")
        pdf.ln(3)

        # --- Generate Alerts Over Time Chart ---
        df_log['Minute'] = pd.to_datetime(df_log['Time']).dt.floor('min')
        df_time = df_log.groupby('Minute').size()
        
        fig, ax = plt.subplots(figsize=(6,3))
        df_time.plot(ax=ax, color='red')
        ax.set_title("Alerts per Minute")
        ax.set_xlabel("Time")
        ax.set_ylabel("Count")
        plt.tight_layout()
        
        # Save chart temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_chart:
            fig.savefig(tmp_chart.name, format="PNG")
            tmp_chart_path = tmp_chart.name
        plt.close(fig)
        pdf.image(tmp_chart_path, w=pdf.w-20, h=60)
        pdf.ln(5)

        # --- Generate Cumulative Heatmap ---
        heatmap = st.session_state["alert_map"]
        if heatmap is not None:
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.imshow(heatmap, cmap='hot')
            ax2.set_title("Cumulative Risk Heatmap")
            ax2.axis('off')
            plt.tight_layout()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_heat:
                fig2.savefig(tmp_heat.name, format="PNG")
                tmp_heat_path = tmp_heat.name
            plt.close(fig2)
            pdf.image(tmp_heat_path, w=pdf.w-20, h=60)
            pdf.ln(5)

        # --- Optional: Include small snapshots ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Sample Alert Snapshots:", ln=True)
        pdf.set_font("Arial", '', 10)
        for i, item in enumerate(st.session_state.violation_images[:5]):  # limit to first 5 for PDF size
            img_data = base64.b64decode(item["image"])
            img = Image.open(io.BytesIO(img_data))
            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
                img.save(tmp_img.name)
                tmp_img_path = tmp_img.name
            pdf.image(tmp_img_path, w=pdf.w-40, h=pdf.w*0.3)
            pdf.ln(5)

        # --- Download PDF (fixed output method) ---
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button(
            L("download_pdf"),
            data=pdf_bytes,
            file_name=f"coastal_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    if st.session_state.violation_images:
        if st.button(L("download_pdf")):
            generate_pdf_report()

with col_main:
    video_display = st.empty()
    stats_display = st.empty()

# Logging helper (modified to dispatch)
def log_alert(frame, reason, extra=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log.append({"Time": timestamp, "Alert": reason, "Details": extra or ""})
    if auto_snapshot:
        try:
            _, buf = cv2.imencode('.jpg', frame)
            encoded = base64.b64encode(buf).decode('utf-8')
            st.session_state.violation_images.append({
                "time": timestamp,
                "reason": reason,
                "image": encoded
            })
        except Exception as e:
            st.sidebar.warning(f"Failed to capture snapshot: {e}")
    st.session_state.alert_count += 1
    st.session_state.last_snapshot = frame
    st.session_state["alert_flag"] = True

    # Prepare message and coordinates (if possible)
    coords = None
    # Attempt to compute approximate centroid from extra
    try:
        if extra and isinstance(extra, dict) and "pos" in extra:
            cx, cy = extra["pos"]
            h, w = frame.shape[:2]
            latlon = pixel_to_latlon(cx, cy, w, h)
            coords = latlon
    except Exception:
        coords = None

    message_text = f"🚨 Urgent Alert: {reason}\nTime: {timestamp}"
    # Attach extra details if present
    if extra:
        message_text += f"\nDetails: {extra}"

    # Dispatch (non-blocking in this script — but may take time)
    try:
        dispatch_alerts(message_text, frame_bgr=frame, approx_coords=coords)
    except Exception as e:
        st.sidebar.error(f"Dispatch failed: {e}")

# === Run Loop ===
if st.session_state.run:
    cap = cv2.VideoCapture(0) if run_source=="Webcam (camera 0)" else None
    if run_source=="Upload video file":
        if uploaded_file is None:
            st.error("Please upload a video file or switch to Webcam.")
            st.session_state.run = False
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.getbuffer())
            cap = cv2.VideoCapture(tfile.name)

    if cap is None or not cap.isOpened():
        st.error("Video source not available.")
        st.session_state.run = False
    else:
        frame_count = 0
        start_time = time.time()
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.sidebar.info("Stream ended or frame not available.")
                break

            st.session_state["alert_flag"] = False
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_time = time.time()
            rock_polys_px = [norm_to_px(p, w, h) for p in rock_polys]
            wave_zone_px = norm_to_px(wave_zone, w, h) if wave_zone else []

            if st.session_state["alert_map"] is None:
                st.session_state["alert_map"] = np.zeros((h,w), dtype=np.uint8)

            # Tide-line simulation
            tide_offset = int(20*np.sin(time.time()/5.0))
            safe_line_y = int(safe_line_frac * h) + tide_offset
            deep_water_y = int(deep_water_frac * h)

            results = model.predict(frame, conf=confidence_threshold, verbose=False)[0]
            detections = []
            centroids = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if label in ["person", "boat", "surfboard"]:
                    cx = int((x1 + x2)/2)
                    cy = int((y1 + y2)/2)
                    detections.append({"label": label, "conf": conf, "box": (x1,y1,x2,y2), "centroid": (cx,cy)})
                    centroids.append((cx,cy))

            update_tracks(centroids, frame_time)
            display = frame.copy()

            # Draw zones
            if wave_zone_px:
                cv2.polylines(display, [np.array(wave_zone_px)], True, (200,200,255), 2)
            for poly in rock_polys_px:
                cv2.polylines(display, [np.array(poly)], True, (0,120,255), 2)
            cv2.line(display, (0, safe_line_y), (w, safe_line_y), (150,255,150), 2)
            cv2.putText(display, "Safe-line", (10, safe_line_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,255,150), 2)
            cv2.line(display, (0, deep_water_y), (w, deep_water_y), (0,150,255), 2)
            cv2.putText(display, "Deep water", (10, deep_water_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,255), 2)

            # Overlay cumulative alert heatmap
            alpha = 0.4
            heatmap_color = cv2.applyColorMap(st.session_state["alert_map"], cv2.COLORMAP_JET)
            cv2.addWeighted(heatmap_color, alpha, display, 1-alpha, 0, display)

            # --- Alerts & tracking logic ---
            persons_in_wave_zone = 0
            persons_near_rocks = []
            deep_water_persons = []
            rip_candidates = []

            for tid, tr in st.session_state["tracks"].items():
                cen = tr["centroid"]
                vx, vy = track_velocity(tr)
                # Draw ID & trajectory
                cv2.circle(display, (int(cen[0]), int(cen[1])), 4, (255,255,0), -1)
                cv2.putText(display, f"ID:{tid}", (int(cen[0]+6), int(cen[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
                if len(tr["trace"]) >= 2:
                    for i in range(len(tr["trace"])-1):
                        cv2.line(display, tr["trace"][i][1], tr["trace"][i+1][1], (255,255,100), 2)

                # Zone checks
                if cen[1] > deep_water_y:
                    deep_water_persons.append((tid, cen, vx, vy))
                for poly in rock_polys_px:
                    if point_in_poly(cen, poly):
                        persons_near_rocks.append((tid, cen))
                if wave_zone_px and point_in_poly(cen, wave_zone_px):
                    persons_in_wave_zone += 1
                if vy > rip_speed_thresh:
                    rip_candidates.append((tid, cen, vx, vy))

            # Rip detection
            rip_alert = False
            rip_details = None
            if len(rip_candidates) >= rip_min_people:
                xs = [c[1][0] for c in rip_candidates]
                span = max(xs) - min(xs) if xs else 0
                if span < w * 0.25:
                    rip_alert = True
                    rip_details = {"count": len(rip_candidates), "span_px": span}

            # Alerts
            if persons_in_wave_zone >= crowd_threshold:
                alert_text = f"High crowding in wave-impact zone ({persons_in_wave_zone})"
                log_alert(display.copy(), alert_text, extra={"count": persons_in_wave_zone})
            for tid, cen in persons_near_rocks:
                alert_text = f"Person near rocks (ID {tid})"
                log_alert(display.copy(), alert_text, extra={"track_id": tid, "pos": cen})
            if rip_alert:
                alert_text = f"Possible rip current detected: {rip_details['count']} persons in channel (span {int(rip_details['span_px'])} px)"
                log_alert(display.copy(), alert_text, extra=rip_details)

            # Deep water stationary detection
            for tid, cen, vx, vy in deep_water_persons:
                speed = math.hypot(vx, vy)
                tr = st.session_state["tracks"].get(tid) or st.session_state["track_history"].get(tid)
                stationary = False
                if tr:
                    radius_px = max(20, int(w*0.03))
                    consec = 0
                    for tstamp, p in reversed(tr["trace"]):
                        if euclid(p, cen) < radius_px:
                            consec += 1
                        else:
                            break
                    if len(tr["trace"]) >= 2 and consec > 0:
                        t0_idx = -consec if consec <= len(tr["trace"]) else 0
                        t0 = tr["trace"][t0_idx][0] if consec <= len(tr["trace"]) else tr["trace"][0][0]
                        t1 = tr["trace"][-1][0]
                        if (t1 - t0) >= stationary_time_thresh and speed < 15:
                            stationary = True
                if stationary:
                    alert_text = f"Stationary person in deep water (possible drowning) ID:{tid}"
                    log_alert(display.copy(), alert_text, extra={"track_id": tid, "pos": cen})

            # Draw detections & update heatmap
            for det in detections:
                x1,y1,x2,y2 = det["box"]
                label = det["label"]
                conf = det["conf"]
                color = (0,255,0) if label=="person" else (255,200,0)
                cv2.rectangle(display, (x1,y1),(x2,y2), color, 2)
                cv2.putText(display, f"{label} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cv2.circle(st.session_state["alert_map"], (cx, cy), 15, 255, -1)

            # FPS
            frame_count += 1
            if frame_time - start_time >= 1.0:
                fps = frame_count / (frame_time - start_time)
                frame_count = 0
                start_time = frame_time
                st.session_state["fps_history"].append(fps)
                if len(st.session_state["fps_history"]) > 50:
                    st.session_state["fps_history"].pop(0)

            if show_debug:
                txt = f"Detections: {len(detections)} | Tracks: {len(st.session_state['tracks'])} | FPS: {st.session_state['fps_history'][-1] if st.session_state['fps_history'] else 0:.1f}"
                cv2.putText(display, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            # --- Play sound once per frame if any alert happened (legacy path)---
            if st.session_state.get("alert_flag", False) and sound_path:
                now = time.time()
                last = st.session_state.get("last_sound_time", 0.0)
                if now - last >= float(sound_cooldown):
                    play_sound_auto(sound_path)
                    st.session_state["last_sound_time"] = now

            video_display.image(display, channels="BGR")
            if show_debug:
                stats_display.line_chart(pd.DataFrame({"fps": st.session_state["fps_history"][-50:]}))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        st.session_state.run = False
else:
    st.info(f"{L('start')} to begin monitoring. Configure zones and parameters on the left. For best results, adjust safe_line & deep_water visually while running.")

st.markdown("---")
st.markdown("Notes: This app integrates tide simulation, alert heatmaps, trajectory prediction, risk visualization, PDF report generation, multi-channel alert dispatch (SMS/WhatsApp/Telegram/Email), voice TTS, and multilingual UI.")
