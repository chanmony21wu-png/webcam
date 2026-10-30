import cv2
import math
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# ---------- load memes ----------
def load(name):
    img = cv2.imread(f"memes/{name}.png")
    if img is None:
        raise FileNotFoundError(f"memes/{name}.png not found or not readable")
    return cv2.resize(img, (480, 480))

memes = {
    "face": load("face"),
    "idle": load("idle"),
    "shhh": load("shhh"),
    "think": load("think"),
}

# ---------- helpers ----------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def fingers_up(hand):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [hand.landmark[t].y < hand.landmark[p].y for t, p in zip(tips, pips)]

def detect_gesture(hand, face):
    up = fingers_up(hand)
    index_tip = hand.landmark[8]

    if face is not None:
        face_width = dist(face.landmark[234], face.landmark[454])
        mouth = face.landmark[13]
        chin = face.landmark[152]

        # SHHH: index finger near the mouth
        if dist(index_tip, mouth) < face_width * 0.25:
            return "shhh"

        # THINK: index finger near the chin
        if dist(index_tip, chin) < face_width * 0.3:
            return "think"

    # FACE: open hand (all four fingers up)
    if all(up):
        return "face"

    return "idle"

# ---------- main loop ----------
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands, \
     mp_face.FaceMesh(max_num_faces=1) as face_mesh:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_res = hands.process(rgb)
        face_res = face_mesh.process(rgb)

        face = face_res.multi_face_landmarks[0] if face_res.multi_face_landmarks else None
        gesture = "idle"

        if hand_res.multi_hand_landmarks:
            hand = hand_res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand, face)

        cv2.putText(frame, f"gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)
        cv2.imshow("Meme", memes.get(gesture, memes["idle"]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()