import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import os

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
labels = joblib.load("labels.pkl")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# ---------------- LANGUAGE ----------------
language = "EN"

translations = {
    "Hello": {"HI": "नमस्ते", "MR": "नमस्कार", "BN": "নমস্কার"},
    "Thanks": {"HI": "धन्यवाद", "MR": "धन्यवाद", "BN": "ধন্যবাদ"},
    "Yes": {"HI": "हाँ", "MR": "होय", "BN": "হ্যাঁ"},
    "No": {"HI": "नहीं", "MR": "नाही", "BN": "না"},
    "ILoveYou": {
        "HI": "मैं तुमसे प्यार करता हूँ",
        "MR": "मी तुझ्यावर प्रेम करतो",
        "BN": "আমি তোমাকে ভালোবাসি"
    }
}

# ---------------- FONTS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

font_hi = ImageFont.truetype(os.path.join(BASE_DIR, "fonts", "NotoSansDevanagari-Regular.ttf"), 32)
font_bn = ImageFont.truetype(os.path.join(BASE_DIR, "fonts", "NotoSansBengali-Regular.ttf"), 32)

# ---------------- HELPERS ----------------
def draw_text(frame, text, position, lang):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    if lang in ["HI", "MR"]:
        draw.text(position, text, font=font_hi, fill=(0,255,120))
    elif lang == "BN":
        draw.text(position, text, font=font_bn, fill=(0,255,120))

    return np.array(img_pil)

def wrap_text(text, max_chars):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        if len(current + word) < max_chars:
            current += word + " "
        else:
            lines.append(current)
            current = word + " "

    lines.append(current)
    return lines

# ---------------- VARIABLES ----------------
sequence = []
sentence = []
predictions = deque(maxlen=10)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Sign Language AI", cv2.WINDOW_NORMAL)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # -------- KEYPOINT EXTRACTION --------
    keypoints = np.zeros(63)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        temp = []
        for lm in hand_landmarks.landmark:
            temp.extend([lm.x, lm.y, lm.z])

        if len(temp) == 63:
            keypoints = temp

        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    sequence.append(keypoints)

    if len(sequence) > 30:
        sequence.pop(0)

    # -------- PREDICTION --------
    if len(sequence) == 30 and all(len(f) == 63 for f in sequence):
        input_data = np.array(sequence).flatten().reshape(1, -1)
        pred = model.predict(input_data)[0]
        predictions.append(pred)

        if predictions.count(pred) > 7:
            word = labels[pred]
            if len(sentence) == 0 or sentence[-1] != word:
                sentence.append(word)

    # -------- UI --------
    h, w, _ = image.shape
    panel_width = int(w * 0.35)

    # Glass effect
    blur = cv2.GaussianBlur(image[:, w-panel_width:w], (25,25), 0)
    panel = blur.copy()

    overlay = panel.copy()
    cv2.rectangle(overlay, (0,0), (panel_width,h), (20,20,20), -1)
    panel = cv2.addWeighted(overlay, 0.6, panel, 0.4, 0)

    # Title
    cv2.putText(panel, "AI Interpreter", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,200), 2)

    # Language
    cv2.putText(panel, f"Lang: {language}", (20,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

    # Current word
    if len(predictions) > 0:
        current = labels[predictions[-1]]

        display_text = current
        if language != "EN" and current in translations:
            display_text = translations[current][language]

        cv2.putText(panel, "CURRENT", (20,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,120), 1)

        if language == "EN":
            cv2.putText(panel, display_text, (20,190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,120), 3)
        else:
            panel = draw_text(panel, display_text, (20,180), language)

    # Confidence bar
    if len(predictions) > 0:
        confidence = predictions.count(predictions[-1]) / len(predictions)

        bar_x, bar_y = 20, 230
        bar_w = int((panel_width - 40) * confidence)

        cv2.rectangle(panel, (bar_x, bar_y), (panel_width-20, bar_y+20), (50,50,50), -1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+20), (0,255,120), -1)

        cv2.putText(panel, f"{int(confidence*100)}%", (bar_x, bar_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    # Sentence
    cv2.putText(panel, "SENTENCE", (20,280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,120), 1)

    full_sentence = " ".join(sentence)

    if language != "EN":
        translated_words = []
        for word in sentence:
            if word in translations:
                translated_words.append(translations[word][language])
            else:
                translated_words.append(word)
        full_sentence = " ".join(translated_words)

    wrapped = wrap_text(full_sentence, max_chars=22)

    y_offset = 320
    for line in wrapped[-5:]:
        if language == "EN":
            cv2.putText(panel, line.strip(), (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            panel = draw_text(panel, line.strip(), (20, y_offset), language)
        y_offset += 35

    # Controls
    cv2.putText(panel, "L Language   C Clear   Q Quit",
                (20, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)

    # Combine
    combined = np.hstack((image, panel))
    cv2.imshow("Sign Language AI", combined)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        sentence = []
    elif key & 0xFF == ord('l'):
        if language == "EN":
            language = "HI"
        elif language == "HI":
            language = "MR"
        elif language == "MR":
            language = "BN"
        else:
            language = "EN"

cap.release()
cv2.destroyAllWindows()