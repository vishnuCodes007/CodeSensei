import ctypes
import os
import queue
import threading
import tkinter as tk
from collections import deque
from datetime import datetime

import cv2
import customtkinter as ctk
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
labels = joblib.load("labels.pkl")

# Minimum model probability (from predict_proba) before a prediction is
# trusted enough to feed the stability buffer / sentence / speech. Without
# this, a 30-frame window of near-zero keypoints (no hand in view) still gets
# classified as one of the trained words with high apparent "stability".
CONFIDENCE_THRESHOLD = 0.6

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# ---------------- LANGUAGES ----------------
LANGUAGE_CODES = {"English": "EN", "Hindi": "HI", "Bengali": "BN", "Tamil": "TA"}

translations = {
    "Hello": {"HI": "नमस्ते", "BN": "নমস্কার", "TA": "வணக்கம்"},
    "Thanks": {"HI": "धन्यवाद", "BN": "ধন্যবাদ", "TA": "நன்றி"},
    "Yes": {"HI": "हाँ", "BN": "হ্যাঁ", "TA": "ஆம்"},
    "No": {"HI": "नहीं", "BN": "না", "TA": "இல்லை"},
    "ILoveYou": {
        "HI": "मैं तुमसे प्यार करता हूँ",
        "BN": "আমি তোমাকে ভালোবাসি",
        "TA": "நான் உன்னை காதலிக்கிறேன்",
    },
    "Please": {"HI": "कृपया", "BN": "দয়া করে", "TA": "தயவுசெய்து"},
    "Sorry": {"HI": "माफ़ करें", "BN": "দুঃখিত", "TA": "மன்னிக்கவும்"},
    "Help": {"HI": "मदद", "BN": "সাহায্য", "TA": "உதவி"},
    "Good": {"HI": "अच्छा", "BN": "ভালো", "TA": "நல்லது"},
    "Stop": {"HI": "रुको", "BN": "থামুন", "TA": "நிறுத்து"},
    "Water": {"HI": "पानी", "BN": "জল", "TA": "தண்ணீர்"},
    "Food": {"HI": "खाना", "BN": "খাবার", "TA": "உணவு"},
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- THEME ----------------
# Deep charcoal neutral palette with one accent color (teal), reserved for the
# confidence meter, the "active"/current prediction, the primary button and
# other active/interactive states. Everything else stays neutral so the
# accent actually stands out.
BG_ROOT = "#1a1a1e"
BG_PANEL = "#212126"
BG_CARD = "#26262c"
BG_INPUT = "#2b2b32"
BORDER = "#35353c"
TEXT_PRIMARY = "#f2f2f0"
TEXT_SECONDARY = "#c6c6cf"
TEXT_MUTED = "#84848f"
ACCENT = "#2dd4bf"
ACCENT_HOVER = "#24b3a0"
ACCENT_ON = "#0d1414"  # text color placed on top of accent-filled surfaces
DANGER = "#e5484d"
DANGER_HOVER = "#c93f44"

# ---------------- FONTS (Inter) ----------------
# Inter isn't installed system-wide, so the bundled static TTFs are loaded as
# a private, process-only font resource (Windows GDI). This makes "Inter" and
# "Inter Medium" addressable by CTkFont without installing anything. Falls
# back to Segoe UI on non-Windows platforms or if loading fails.
INTER_DIR = os.path.join(BASE_DIR, "fonts", "inter")
_FR_PRIVATE = 0x10


def _register_app_fonts():
    if os.name != "nt":
        return False
    try:
        gdi32 = ctypes.windll.gdi32
        loaded_any = False
        for weight in ("Regular", "Medium", "SemiBold", "Bold"):
            path = os.path.join(INTER_DIR, f"Inter-{weight}.ttf")
            if os.path.exists(path) and gdi32.AddFontResourceExW(path, _FR_PRIVATE, 0):
                loaded_any = True
        return loaded_any
    except Exception:
        return False


_INTER_AVAILABLE = _register_app_fonts()
FONT_FAMILY = "Inter" if _INTER_AVAILABLE else "Segoe UI"
FONT_FAMILY_MEDIUM = "Inter Medium" if _INTER_AVAILABLE else "Segoe UI"


def font_heading(size=20):
    """Bold tier: headings."""
    return ctk.CTkFont(family=FONT_FAMILY, size=size, weight="bold")


def font_label(size=14):
    """Medium tier: labels/buttons."""
    return ctk.CTkFont(family=FONT_FAMILY_MEDIUM, size=size)


def font_caption(size=12):
    """Regular tier: secondary/caption text (timestamps, hints)."""
    return ctk.CTkFont(family=FONT_FAMILY, size=size)


# ---------------- INDIC (Hindi/Bengali/Tamil) FONT RENDERING ----------------
# Nirmala UI ships with Windows 8+ and is a single pan-Indic font covering
# Devanagari, Bengali and Tamil correctly. Fall back to the bundled
# Devanagari-only font if it isn't present (e.g. a non-Windows machine).
SYSTEM_INDIC_FONT = r"C:\Windows\Fonts\Nirmala.ttc"
FALLBACK_INDIC_FONT = os.path.join(BASE_DIR, "fonts", "NotoSansDevanagari-Regular.ttf")
INDIC_TEXT_FILL = (45, 212, 191, 255)  # matches ACCENT


def _load_indic_font(size):
    if os.path.exists(SYSTEM_INDIC_FONT):
        return ImageFont.truetype(SYSTEM_INDIC_FONT, size, index=0)
    return ImageFont.truetype(FALLBACK_INDIC_FONT, size)


def render_indic_text(text, font, fill=INDIC_TEXT_FILL):
    """Render text to an RGBA PIL image using Pillow, since Tk/cv2 text
    rendering cannot reliably shape Devanagari/Bengali/Tamil glyphs."""
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = max(1, bbox[2] - bbox[0] + 6)
    h = max(1, bbox[3] - bbox[1] + 6)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0] + 3, -bbox[1] + 3), text, font=font, fill=fill)
    return img


# ---------------- WORDMARK / APP ICON ----------------
def render_wordmark_icon(size=32, accent=ACCENT, mark_color=BG_ROOT):
    """Simple geometric mark: a rounded accent badge with an abstract 3-bar
    signal/soundwave glyph, standing in for both "sign" (visual) and "speak"
    (audio)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=size * 0.30, fill=accent)

    bar_w = max(2, round(size * 0.12))
    gap = max(2, round(size * 0.10))
    heights = [size * 0.30, size * 0.52, size * 0.38]
    total_w = bar_w * 3 + gap * 2
    x = (size - total_w) / 2
    for h in heights:
        y0 = (size - h) / 2
        y1 = y0 + h
        draw.rounded_rectangle([x, y0, x + bar_w, y1], radius=bar_w / 2, fill=mark_color)
        x += bar_w + gap
    return img


def _write_window_icon():
    icon_img = render_wordmark_icon(size=256)
    bg = Image.new("RGBA", icon_img.size, BG_ROOT)
    composed = Image.alpha_composite(bg, icon_img)
    path = os.path.join(BASE_DIR, "app_icon.ico")
    composed.save(path, sizes=[(16, 16), (32, 32), (48, 48), (128, 128), (256, 256)])
    return path


# ---------------- TEXT-TO-SPEECH ----------------
# Hints used to match an installed SAPI5 voice to a language, if the user has
# added the matching Windows Speech language pack (Settings > Time & Language
# > Speech > Add a voice). Without a matching voice installed, speech falls
# back to reading the English word so there is still audible feedback.
VOICE_LANGUAGE_HINTS = {
    "HI": ["hindi", "hi-in", "hi_in"],
    "BN": ["bengali", "bangla", "bn-in", "bn_in"],
    "TA": ["tamil", "ta-in", "ta_in"],
}


class TTSWorker:
    """Runs pyttsx3 on a dedicated thread so speech never blocks the UI loop.

    pyttsx3's Windows driver talks to SAPI5 through COM, and COM objects are
    apartment-threaded: an engine created on one thread cannot be driven from
    another. So the engine must be constructed *inside* the worker thread
    (not in __init__, which runs on the Tk main thread) or every say()/
    runAndWait() call silently does nothing.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _discover_voices(self, engine):
        mapping = {}
        for voice in engine.getProperty("voices"):
            haystack = f"{voice.id} {voice.name} {' '.join(getattr(voice, 'languages', None) or [])}".lower()
            for lang, hints in VOICE_LANGUAGE_HINTS.items():
                if lang not in mapping and any(hint in haystack for hint in hints):
                    mapping[lang] = voice.id
        return mapping

    def _run(self):
        engine = pyttsx3.init()
        default_voice = engine.getProperty("voice")
        voice_by_lang = self._discover_voices(engine)

        while True:
            item = self._queue.get()
            if item is None:
                break
            text, lang, fallback_text = item
            voice_id = voice_by_lang.get(lang)
            if voice_id:
                engine.setProperty("voice", voice_id)
                engine.say(text)
            else:
                engine.setProperty("voice", default_voice)
                engine.say(fallback_text)
            engine.runAndWait()

    def speak(self, text, lang="EN", fallback_text=None):
        self._queue.put((text, lang, fallback_text if fallback_text is not None else text))

    def stop(self):
        self._queue.put(None)


# ---------------- APP ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SignSpeak")
        try:
            self.iconbitmap(_write_window_icon())
        except Exception:
            pass

        self.geometry("1180x680")
        self.minsize(980, 580)
        self.configure(fg_color=BG_ROOT)

        self.language = "EN"
        self.sequence = []
        self.sentence = []
        self.history = []
        self.predictions = deque(maxlen=10)

        self._conf_anim_job = None
        self._flash_anim_job = None

        self.tts = TTSWorker()
        self.cap = cv2.VideoCapture(0)
        self._current_frame_image = None

        self._prerender_translated_words()
        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.after(10, self.update_frame)

    # ---------------- translated-word pre-rendering ----------------
    def _prerender_translated_words(self):
        font_large = _load_indic_font(30)
        font_word = _load_indic_font(18)
        font_small = _load_indic_font(14)

        # (lang, word) -> {"large": CTkImage, "small": CTkImage}
        self.translated_ctk_images = {}
        # (lang, word) -> ImageTk.PhotoImage, for embedding inline in tk.Text
        self.translated_photo_images = {}

        for word, tr in translations.items():
            for lang, text in tr.items():
                img_large = render_indic_text(text, font_large)
                img_small = render_indic_text(text, font_small)
                self.translated_ctk_images[(lang, word)] = {
                    "large": ctk.CTkImage(light_image=img_large, dark_image=img_large, size=img_large.size),
                    "small": ctk.CTkImage(light_image=img_small, dark_image=img_small, size=img_small.size),
                }
                self.translated_photo_images[(lang, word)] = ImageTk.PhotoImage(
                    render_indic_text(text, font_word)
                )

    def display_text_for(self, word):
        if self.language != "EN" and self.language in translations.get(word, {}):
            return translations[word][self.language]
        return word

    # ---------------- UI construction ----------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        PAD = 22

        # ---- video panel ----
        video_frame = ctk.CTkFrame(self, corner_radius=8, fg_color=BG_PANEL)
        video_frame.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(video_frame, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # ---- sidebar ----
        sidebar = ctk.CTkFrame(self, corner_radius=16, fg_color=BG_PANEL)
        sidebar.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        r = 0

        # wordmark
        wordmark_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        wordmark_row.grid(row=r, column=0, sticky="w", padx=PAD, pady=(PAD, 2))
        r += 1
        icon_img = render_wordmark_icon(size=30)
        self._wordmark_icon = ctk.CTkImage(light_image=icon_img, dark_image=icon_img, size=(30, 30))
        ctk.CTkLabel(wordmark_row, image=self._wordmark_icon, text="").pack(side="left")
        ctk.CTkLabel(
            wordmark_row, text="SignSpeak", font=font_heading(20), text_color=TEXT_PRIMARY
        ).pack(side="left", padx=(10, 0))

        self._add_divider(sidebar, r, top=12, bottom=16)
        r += 1

        # language dropdown
        lang_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        lang_row.grid(row=r, column=0, sticky="ew", padx=PAD, pady=(0, 16))
        r += 1
        ctk.CTkLabel(lang_row, text="Language", font=font_label(14), text_color=TEXT_SECONDARY).pack(side="left")
        self.lang_menu = ctk.CTkOptionMenu(
            lang_row, values=list(LANGUAGE_CODES.keys()), command=self._on_language_change,
            width=130, corner_radius=10, font=font_label(13),
            fg_color=BG_INPUT, button_color=BORDER, button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=BG_CARD, dropdown_hover_color=ACCENT, dropdown_text_color=TEXT_PRIMARY,
            text_color=TEXT_PRIMARY,
        )
        self.lang_menu.set("English")
        self.lang_menu.pack(side="right")

        # current prediction card
        cur_frame = ctk.CTkFrame(sidebar, corner_radius=14, fg_color=BG_CARD)
        cur_frame.grid(row=r, column=0, sticky="ew", padx=PAD, pady=(0, 14))
        r += 1
        cur_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            cur_frame, text="CURRENT", text_color=TEXT_MUTED, font=font_caption(12)
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 0))
        self.current_word_label = ctk.CTkLabel(
            cur_frame, text="—", font=font_heading(26), text_color=ACCENT
        )
        self.current_word_label.grid(row=1, column=0, sticky="w", padx=16, pady=(2, 16))

        # confidence meter
        conf_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        conf_frame.grid(row=r, column=0, sticky="ew", padx=PAD, pady=(0, 18))
        r += 1
        conf_frame.grid_columnconfigure(0, weight=1)
        self.conf_bar = ctk.CTkProgressBar(conf_frame, progress_color=ACCENT, fg_color=BG_INPUT, height=8)
        self.conf_bar.set(0)
        self.conf_bar.grid(row=0, column=0, sticky="ew")
        self.conf_label = ctk.CTkLabel(conf_frame, text="0%", width=40, font=font_caption(12), text_color=TEXT_MUTED)
        self.conf_label.grid(row=0, column=1, padx=(10, 0))

        self._add_divider(sidebar, r)
        r += 1

        # sentence builder
        ctk.CTkLabel(
            sidebar, text="SENTENCE", text_color=TEXT_MUTED, font=font_caption(12)
        ).grid(row=r, column=0, sticky="w", padx=PAD, pady=(14, 6))
        r += 1

        sentence_card = ctk.CTkFrame(sidebar, corner_radius=14, fg_color=BG_CARD)
        sentence_card.grid(row=r, column=0, sticky="nsew", padx=PAD, pady=(0, 14))
        sidebar.grid_rowconfigure(r, weight=2)
        r += 1
        sentence_card.grid_rowconfigure(0, weight=1)
        sentence_card.grid_columnconfigure(0, weight=1)

        self.sentence_textbox = tk.Text(
            sentence_card, wrap="word", bg=BG_CARD, fg=TEXT_PRIMARY, bd=0,
            highlightthickness=0, font=(FONT_FAMILY, 13), padx=14, pady=14,
            insertbackground=TEXT_PRIMARY,
        )
        self.sentence_textbox.grid(row=0, column=0, sticky="nsew")
        sentence_scroll = ctk.CTkScrollbar(
            sentence_card, command=self.sentence_textbox.yview,
            button_color=BORDER, button_hover_color=TEXT_MUTED,
        )
        sentence_scroll.grid(row=0, column=1, sticky="ns", pady=8, padx=(0, 8))
        self.sentence_textbox.configure(yscrollcommand=sentence_scroll.set, state="disabled")

        self._add_divider(sidebar, r)
        r += 1

        # history log
        ctk.CTkLabel(
            sidebar, text="HISTORY", text_color=TEXT_MUTED, font=font_caption(12)
        ).grid(row=r, column=0, sticky="w", padx=PAD, pady=(14, 6))
        r += 1

        self.history_frame = ctk.CTkScrollableFrame(
            sidebar, corner_radius=14, fg_color=BG_CARD,
            scrollbar_button_color=BORDER, scrollbar_button_hover_color=TEXT_MUTED,
        )
        self.history_frame.grid(row=r, column=0, sticky="nsew", padx=PAD, pady=(0, 16))
        sidebar.grid_rowconfigure(r, weight=3)
        r += 1
        self.history_frame.grid_columnconfigure(0, weight=1)

        # buttons
        btn_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_row.grid(row=r, column=0, sticky="ew", padx=PAD, pady=(0, PAD))
        r += 1
        btn_row.grid_columnconfigure((0, 1), weight=1)

        clear_btn = ctk.CTkButton(
            btn_row, text="Clear", corner_radius=14, command=self.on_clear,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=ACCENT_ON,
            font=font_label(14), border_width=0,
        )
        clear_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._add_button_feedback(clear_btn, ACCENT)

        quit_btn = ctk.CTkButton(
            btn_row, text="Quit", corner_radius=14, fg_color=BG_INPUT,
            hover_color=DANGER_HOVER, text_color=TEXT_SECONDARY,
            font=font_label(14), border_width=0, command=self.on_quit,
        )
        quit_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._add_button_feedback(quit_btn, DANGER)

    def _add_divider(self, parent, row, top=10, bottom=10):
        divider = ctk.CTkFrame(parent, height=1, fg_color=BORDER)
        divider.grid(row=row, column=0, sticky="ew", padx=22, pady=(top, bottom))

    def _add_button_feedback(self, button, glow_color, base_border=0, hover_border=2, press_border=3):
        """Hover/pressed feedback beyond CTk's built-in hover_color: a border
        glow ring that appears on hover and thickens on press."""
        button.configure(border_width=base_border)

        def on_enter(_e):
            button.configure(border_width=hover_border, border_color=glow_color)

        def on_leave(_e):
            button.configure(border_width=base_border)

        def on_press(_e):
            button.configure(border_width=press_border, border_color=glow_color)

        def on_release(_e):
            button.configure(border_width=hover_border, border_color=glow_color)

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        button.bind("<ButtonPress-1>", on_press)
        button.bind("<ButtonRelease-1>", on_release)

    # ---------------- language handling ----------------
    def _on_language_change(self, value):
        self.language = LANGUAGE_CODES.get(value, "EN")
        self._rebuild_sentence_view()
        self._rebuild_history_view()
        if self.predictions:
            self._update_current_word_display(labels[self.predictions[-1]])

    # ---------------- current word / confidence ----------------
    def _update_current_word_display(self, word):
        if word is None:
            self.current_word_label.configure(text="—", image=None)
            self.current_word_label._label.configure(image="")
            return
        key = (self.language, word)
        if self.language != "EN" and key in self.translated_ctk_images:
            self.current_word_label.configure(image=self.translated_ctk_images[key]["large"], text="")
        else:
            self.current_word_label.configure(text=word, image=None)
            # CustomTkinter 6.0 doesn't clear the underlying Tk label's image
            # when configure(image=None) is called, so do it directly to avoid
            # a stale Devanagari image lingering under the new English text.
            self.current_word_label._label.configure(image="")

    def _update_confidence(self, confidence):
        self.conf_label.configure(text=f"{int(confidence * 100)}%")
        self._animate_confidence_bar(confidence)

    def _animate_confidence_bar(self, target, steps=10, interval=16):
        """Ease the progress bar toward `target` instead of snapping, so
        prediction-to-prediction changes read as a smooth transition."""
        if self._conf_anim_job is not None:
            self.after_cancel(self._conf_anim_job)
            self._conf_anim_job = None

        start = self.conf_bar.get()
        diff = target - start

        def step(i=1):
            self.conf_bar.set(start + diff * (i / steps))
            if i < steps:
                self._conf_anim_job = self.after(interval, lambda: step(i + 1))
            else:
                self.conf_bar.set(target)
                self._conf_anim_job = None

        step()

    # ---------------- sentence builder ----------------
    def _rebuild_sentence_view(self, highlight_last=False):
        self.sentence_textbox.configure(state="normal")
        self.sentence_textbox.delete("1.0", "end")
        last_start = None
        for i, word in enumerate(self.sentence):
            if i > 0:
                self.sentence_textbox.insert("end", "   ")
            if i == len(self.sentence) - 1:
                last_start = self.sentence_textbox.index("end-1c")
            key = (self.language, word)
            if self.language != "EN" and key in self.translated_photo_images:
                self.sentence_textbox.image_create("end", image=self.translated_photo_images[key])
            else:
                self.sentence_textbox.insert("end", word)
        self.sentence_textbox.see("end")
        self.sentence_textbox.configure(state="disabled")

        if highlight_last and last_start is not None:
            self.sentence_textbox.tag_remove("flash", "1.0", "end")
            self.sentence_textbox.tag_add("flash", last_start, "end")
            self._animate_flash_tag()

    def _animate_flash_tag(self, steps=14, interval=30):
        """Fade the newly appended word's background from accent to the
        card color, drawing the eye to what just got recognized."""
        if self._flash_anim_job is not None:
            self.after_cancel(self._flash_anim_job)
            self._flash_anim_job = None

        start_rgb = self.winfo_rgb(ACCENT)
        end_rgb = self.winfo_rgb(BG_CARD)

        def step(i=0):
            t = i / steps
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t) >> 8
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t) >> 8
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t) >> 8
            self.sentence_textbox.tag_configure("flash", background=f"#{r:02x}{g:02x}{b:02x}")
            if i < steps:
                self._flash_anim_job = self.after(interval, lambda: step(i + 1))
            else:
                self.sentence_textbox.tag_configure("flash", background=BG_CARD)
                self._flash_anim_job = None

        step()

    # ---------------- history log ----------------
    def _append_history_row(self, ts, word):
        row = ctk.CTkFrame(self.history_frame, fg_color="transparent")
        row.pack(fill="x", pady=4, padx=4)
        ctk.CTkLabel(
            row, text=ts, text_color=TEXT_MUTED, font=font_caption(11), width=64, anchor="w"
        ).pack(side="left")
        key = (self.language, word)
        if self.language != "EN" and key in self.translated_ctk_images:
            ctk.CTkLabel(row, image=self.translated_ctk_images[key]["small"], text="").pack(
                side="left", padx=(8, 0)
            )
        else:
            ctk.CTkLabel(
                row, text=word, anchor="w", font=font_label(13), text_color=TEXT_SECONDARY
            ).pack(side="left", padx=(8, 0))

    def _rebuild_history_view(self):
        for child in self.history_frame.winfo_children():
            child.destroy()
        for ts, word in self.history:
            self._append_history_row(ts, word)
        self.after_idle(lambda: self.history_frame._parent_canvas.yview_moveto(1.0))

    def _add_history_entry(self, word):
        ts = datetime.now().strftime("%H:%M:%S")
        self.history.append((ts, word))
        self._append_history_row(ts, word)
        self.after_idle(lambda: self.history_frame._parent_canvas.yview_moveto(1.0))

    # ---------------- video rendering ----------------
    def _render_video_frame(self, rgb_image):
        pil_img = Image.fromarray(rgb_image)

        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()
        if target_w < 10 or target_h < 10:
            target_w, target_h = 640, 480

        src_w, src_h = pil_img.size
        scale = min(target_w / src_w, target_h / src_h)
        new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
        pil_img = pil_img.resize(new_size)

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=new_size)
        self._current_frame_image = ctk_img
        self.video_label.configure(image=ctk_img, text="")

    # ---------------- buttons ----------------
    def on_clear(self):
        self.sentence = []
        self._rebuild_sentence_view()

    def on_quit(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.tts.stop()
        self.destroy()

    # ---------------- main prediction loop ----------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.after(15, self.update_frame)
            return

        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # -------- KEYPOINT EXTRACTION --------
        hand_present = bool(results.multi_hand_landmarks)

        if hand_present:
            hand_landmarks = results.multi_hand_landmarks[0]
            temp = []
            for lm in hand_landmarks.landmark:
                temp.extend([lm.x, lm.y, lm.z])

            keypoints = temp if len(temp) == 63 else np.zeros(63)

            self.sequence.append(keypoints)
            if len(self.sequence) > 30:
                self.sequence.pop(0)
        else:
            # No hand in frame: drop the in-progress window instead of padding
            # it with zeros, which the model would otherwise confidently (and
            # wrongly) classify as one of the trained words.
            self.sequence = []
            self.predictions.clear()

        # -------- PREDICTION --------
        if hand_present and len(self.sequence) == 30 and all(len(f) == 63 for f in self.sequence):
            input_data = np.array(self.sequence).flatten().reshape(1, -1)
            proba = model.predict_proba(input_data)[0]
            pred = int(np.argmax(proba))
            confidence = float(proba[pred])
            word = labels[pred]

            self._update_current_word_display(word)
            self._update_confidence(confidence)

            if confidence >= CONFIDENCE_THRESHOLD:
                self.predictions.append(pred)
                if self.predictions.count(pred) > 7:
                    if len(self.sentence) == 0 or self.sentence[-1] != word:
                        self.sentence.append(word)
                        self._add_history_entry(word)
                        self._rebuild_sentence_view(highlight_last=True)
                        self.tts.speak(self.display_text_for(word), self.language, fallback_text=word)
        elif not hand_present:
            self._update_current_word_display(None)
            self._update_confidence(0.0)

        # -------- VIDEO --------
        self._render_video_frame(image)

        self.after(15, self.update_frame)


if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()
