import cv2
import numpy as np
from tensorflow.keras.models import load_model
import speech_recognition as sr
import threading
import joblib
import csv
import os
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mediapipe as mp
import pandas as pd
import librosa
import tempfile
import time

fusion_model = load_model('C:/Users/Singh/Desktop/SEM 6 - MINI/fusion_model.keras')
tokenizer = joblib.load('C:/Users/Singh/Desktop/SEM 6 - MINI/tokenizer.pkl')
scaler = joblib.load('C:/Users/Singh/Desktop/SEM 6 - MINI/behavior_scaler.pkl')

feature_names = [
    "facial_expressions", "mouth_movement", "head_movement",
    "gaze_direction", "gestures", "speech_irregularities", "Gender"
]

captured_text = ""
detection_result = "Waiting for input..."
confidence_score = 0.0
saliency_map = {}
behavior_input_global = np.zeros((1, 7))
blink_counter = 0
blink_frames = 0
speech_irregularities = 0.0

def eye_aspect_ratio(eye_pts):
    return np.linalg.norm(eye_pts[1] - eye_pts[5]) / (2.0 * np.linalg.norm(eye_pts[0] - eye_pts[3]) + 1e-6)

def preprocess_text(text, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

def capture_audio():
    global captured_text, speech_irregularities
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for audio input...")
        try:
            audio = recognizer.listen(source, timeout=5)
            captured_text = recognizer.recognize_google(audio)
            print(f"Captured statement: {captured_text}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.get_wav_data())
                y, sr_ = librosa.load(tmp.name, sr=None)
                rms = librosa.feature.rms(y=y).mean()
                pitch = librosa.yin(y, fmin=50, fmax=300).std()
                captured_text_lower = captured_text.lower()
                fillers = sum(captured_text_lower.count(word) for word in ["um", "uh", "like"])
                speech_irregularities = min(1.0, 0.5 * fillers + 0.25 * pitch + 0.25 * rms)
        except:
            print("Could not understand audio")
            speech_irregularities = 0.0

def capture_video():
    global detection_result, confidence_score, saliency_map, behavior_input_global, blink_counter, blink_frames, speech_irregularities
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
    mp_hands = mp.solutions.hands.Hands()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_result = mp_face_mesh.process(rgb)
            hand_result = mp_hands.process(rgb)

            facial_expressions = mouth_movement = head_movement = gaze_direction = 0
            gestures = 0
            gender = 1.0

            if face_result.multi_face_landmarks:
                face = face_result.multi_face_landmarks[0]
                lm = np.array([[p.x, p.y, p.z] for p in face.landmark])

                face_width = np.linalg.norm(lm[234] - lm[454]) + 1e-6
                mouth_open = np.linalg.norm(lm[13] - lm[14]) / face_width
                eyebrow_lift = np.linalg.norm(lm[70] - lm[300]) / face_width
                nose_tip = lm[1][1]
                forehead = lm[10][1]
                head_movement = (forehead - nose_tip) / face_width
                gaze_direction = lm[1][0]

                left_eye = lm[[33, 160, 158, 133, 153, 144]]
                right_eye = lm[[362, 385, 387, 263, 373, 380]]
                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)

                blink_frames += 1
                if ear_left < 0.2 and ear_right < 0.2:
                    blink_counter += 1

                blink_ratio = blink_counter / (blink_frames + 1e-6)
                facial_expressions = eyebrow_lift + blink_ratio

            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
                    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
                    dist = np.linalg.norm(wrist - thumb_tip)
                    if dist < 0.05:
                        gestures = 1

            row = pd.DataFrame([{
                "facial_expressions": facial_expressions,
                "mouth_movement": mouth_open,
                "head_movement": head_movement,
                "gaze_direction": gaze_direction,
                "gestures": gestures,
                "speech_irregularities": speech_irregularities,
                "Gender": gender
            }])
            behavior_input_global = scaler.transform(row)

            label = f"{detection_result} ({confidence_score:.2f}%)"
            color = (0, 255, 0) if detection_result == 'Truth' else (0, 0, 255)
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            y = 90
            for key, val in saliency_map.items():
                try:
                    txt = f"{key}: {float(val):.2f}"
                    cv2.putText(frame, txt, (50, y), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
                    y += 20
                except:
                    continue

            cv2.imshow('Real-Time Deception Detection', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def log_prediction(text, pred, label, saliency_map):
    log_path = "C:/Users/Singh/Desktop/SEM 6 - MINI/predictions_log.csv"
    write_header = not os.path.exists(log_path)

    with open(log_path, "a", newline="") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow(["Timestamp", "Text", "Confidence", "Label"] + feature_names)

        saliency_values = [round(saliency_map.get(feat, 0), 4) for feat in feature_names]
        writer.writerow([datetime.now(), text, round(pred, 4), label] + saliency_values)

def generate_saliency(behavior_input):
    global saliency_map
    try:
        layer = fusion_model.get_layer("behavior_features")
        weights = layer.get_weights()[0]
        feature_scores = np.mean(np.abs(weights), axis=1)  

        input_values = behavior_input.flatten()
        weighted_input = input_values * feature_scores

        total = np.sum(weighted_input)
        if total > 0:
            normalized = weighted_input / total
        else:
            normalized = np.zeros_like(weighted_input)

        saliency_map = dict(zip(feature_names, normalized))

        print("Behavior Input:", input_values)
        print("Feature Scores (mean abs weights):", feature_scores)
        print("Weighted Input:", weighted_input)
        print("Normalized Saliency:", normalized)

    except Exception as e:
        print(f"[Saliency Error]: {e}")
        saliency_map = {"saliency_error": str(e)}

#  MAIN
video_thread = threading.Thread(target=capture_video)
video_thread.start()

try:
    while video_thread.is_alive():
        if captured_text == "":
            audio_thread = threading.Thread(target=capture_audio)
            audio_thread.start()
            audio_thread.join()

        if captured_text:
            text_input = preprocess_text(captured_text)
            behavior_input = behavior_input_global

            pred = fusion_model.predict([text_input, behavior_input])[0][0]
            confidence_score = pred * 100

            speech_weight = behavior_input[0][5]
            threshold_high = 0.6 - 0.2 * speech_weight
            threshold_low = 0.4 - 0.2 * speech_weight

            if pred > threshold_high:
                detection_result = 'Truth'
            elif pred < threshold_low:
                detection_result = 'Lie'
            else:
                detection_result = 'Uncertain'

            generate_saliency(behavior_input)
            log_prediction(captured_text, pred, detection_result, saliency_map)
            print(f'Deception Detection Result: {detection_result} ({confidence_score:.2f}%)')
            captured_text = ""

        time.sleep(0.5)
finally:
    video_thread.join()
