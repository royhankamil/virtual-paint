import cv2
import mediapipe as mp

# Fungsi untuk mendapatkan label gerakan jari
# (fungsi get_finger_count tidak diubah)

# Fungsi untuk mendapatkan label gerakan jari
def get_finger_count(landmarks):
    thumb = landmarks.landmark[4].y > landmarks.landmark[3].y
    index_finger = landmarks.landmark[8].y > landmarks.landmark[7].y
    middle_finger = landmarks.landmark[12].y > landmarks.landmark[11].y
    ring_finger = landmarks.landmark[16].y > landmarks.landmark[15].y
    little_finger = landmarks.landmark[20].y > landmarks.landmark[19].y

    fingers = [thumb, index_finger, middle_finger, ring_finger, little_finger]
    count = fingers.count(True)

    if count == 0:
        return "Nol"
    elif count == 1:
        if thumb:
            return "Jempol"
        else:
            return "Satu"
    elif count == 2:
        if thumb and index_finger:
            return "Dua"
        elif not thumb and index_finger:
            return "Satu"
        else:
            return "Dua"
    elif count == 3:
        if thumb and index_finger and middle_finger:
            return "Tiga"
        elif not thumb and index_finger and middle_finger:
            return "Dua"
        elif not thumb and not index_finger and middle_finger:
            return "Satu"
        else:
            return "Tiga"
    elif count == 4:
        if thumb and index_finger and middle_finger and ring_finger:
            return "Empat"
        elif not thumb and index_finger and middle_finger and ring_finger:
            return "Tiga"
        elif not thumb and not index_finger and middle_finger and ring_finger:
            return "Dua"
        elif not thumb and not index_finger and not middle_finger and ring_finger:
            return "Satu"
        else:
            return "Empat"
    elif count == 5:
        return "Lima"
    else:
        return "Tidak Diketahui"

# Fungsi untuk menggambar landmarks, garis antara titik, dan label gerakan jari
def draw_landmarks_with_label(frame, landmarks):
    connections = mp.solutions.hands.HAND_CONNECTIONS
    mp.solutions.drawing_utils.draw_landmarks(
        frame, landmarks, connections)

    label = get_finger_count(landmarks)
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Fungsi untuk menggambar pada layar
def draw_on_canvas(frame, drawing_points, colors, current_color, drawing):
    if drawing:
        for point in drawing_points:
            cv2.circle(frame, point, 10, colors[current_color], -1)

    return frame

# Inisialisasi modul Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Warna yang tersedia untuk virtual paint
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Misalnya: Merah, Hijau, Biru
current_color = 0
drawing = False

# List untuk menyimpan titik-titik yang digambar
drawing_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Check if landmarks are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get finger count and use it for paint functionality
            finger_count = get_finger_count(landmarks)

            if finger_count == "Empat":  # Deteksi empat jari untuk menggambar
                drawing = True
                h, w, c = frame.shape
                x, y = int(landmarks.landmark[8].x * w), int(landmarks.landmark[8].y * h)  # Gunakan ujung jari telunjuk
                drawing_points.append((x, y))
            else:
                drawing = False

            # Draw on canvas
            frame = draw_on_canvas(frame, drawing_points, colors, current_color, drawing)

    # Display the frame
    cv2.imshow('Virtual Paint', frame)

    # Keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Keluar dari program
        break
    elif key == ord('c'):  # Ganti warna (cycle through available colors)
        current_color = (current_color + 1) % len(colors)
    elif key == ord('e'):  # Hapus gambar (erase canvas)
        drawing_points = []  # Mengosongkan titik-titik yang digambar

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

print()