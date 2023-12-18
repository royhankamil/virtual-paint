import cv2
import mediapipe as mp

# Fungsi untuk menggambar landmarks dan garis antara titik
def draw_landmarks_and_connections(frame, landmarks):
    connections = mp.solutions.hands.HAND_CONNECTIONS
    for landmark in landmarks.landmark:
        h, w, c = frame.shape
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    mp.solutions.drawing_utils.draw_landmarks(
        frame, landmarks, connections)

# Inisialisasi modul Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

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
            draw_landmarks_and_connections(frame, landmarks)

    # Display the frame
    cv2.imshow('Hand Tracking with Connections', frame)

    # Exit the program when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
