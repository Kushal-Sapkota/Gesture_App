import cv2, mediapipe as mp, numpy as np, csv, os
CSV_PATH = "dataset/hand_gesture_landmarks.csv"
GESTURE_KEYS = {'o':'OPEN_PALM','f':'FIST','p':'PINCH','v':'V_SIGN'}
os.makedirs("dataset", exist_ok=True)

def normalize_landmarks(lm_list):
    pts = np.array([[p.x,p.y,p.z] for p in lm_list],dtype=np.float32)
    pts -= pts[0]
    ref = np.linalg.norm(pts[9] - pts[0]) or 1.0
    pts /= ref
    return pts.flatten().tolist()

mp_hands = mp.solutions.hands
with mp_hands.Hands(min_detection_confidence=0.7) as hands, \
     open(CSV_PATH,'a',newline='') as f:
    writer = csv.writer(f)
    if os.stat(CSV_PATH).st_size==0:
        writer.writerow([f"f{i}" for i in range(63)]+["label"])
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        rgb   = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, "Press o/f/p/v to record, ESC to quit",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1)&0xFF
        if key==27: break
        ch = chr(key) if 0<=key<256 else ""
        if ch in GESTURE_KEYS and res.multi_hand_landmarks:
            feats = normalize_landmarks(lm)
            writer.writerow(feats + [GESTURE_KEYS[ch]])
            print("Recorded", GESTURE_KEYS[ch])
    cap.release()
    cv2.destroyAllWindows()
