from fer import FER
import cv2
import matplotlib.pyplot as plt
import os

# Map FER emotions to your emoji filenames
emoji_map = {
    "angry": "emojis/5862951a3796e30ac4468737.png",                        # 😠
    "happy": "emojis/586295b33796e30ac4468738.png",                        # 😊
    "laughing": "emojis/586294383796e30ac4468730.png",                     # 😆
    "surprise": "emojis/58e8ff52eb97430e819064cf.png",                     # 🤡
    "fear": "emojis/580b57fcd9996e24bc43c4af.png",                         # 😱
    "sad": "emojis/5a553a1dedcceab89d2a2de1.png",                          # 😔
    "neutral": "emojis/26570e31-5d90-436d-aac6-7b425a02e892.jpeg",         # 😐
    "disgust": "emojis/5db6d7bd-7712-49da-aa3e-75d185776c08.jpeg"          # ☠️
}

# Load emotion detector
emotion_detector = FER(mtcnn=True)

# Open camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    result = emotion_detector.detect_emotions(frame)

    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        print(f"Detected emotion: {top_emotion}")

        # Get corresponding emoji path
        emoji_path = emoji_map.get(top_emotion, "emojis/58e8ff52eb97430e819064cf.png")  # clown fallback

        if not os.path.exists(emoji_path):
            print(f"⚠️ Emoji not found: {emoji_path}")
            continue

        emoji = cv2.imread(emoji_path)
        if emoji is None:
            print(f"❌ Failed to load emoji image: {emoji_path}")
            continue

        # Resize and overlay emoji
        emoji = cv2.resize(emoji, (w, h))
        try:
            frame[y:y+h, x:x+w] = emoji
        except:
            print("⚠️ Could not overlay emoji — out of frame bounds")

    # Display result
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("❌ Could not access camera")
