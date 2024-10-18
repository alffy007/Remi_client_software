from Remi_eyes.remi_eyes import EmotionDetector
from Remi_ear.remi_ear import user_chatbot_conversation
import threading
import time

def run_emotion_detection(emotion_detector):
    try:
        for emotion in emotion_detector.detect_emotions():
            if emotion == "Failed to capture image":
                print("Camera error!")
            else:
                print(f"Detected Emotion: {emotion}")
                # Handle the detected emotion (e.g., update the UI, log emotions, etc.)
                
            time.sleep(1)  # Add a small delay if necessary to avoid overwhelming the console
    except Exception as e:
        print(f"Emotion detection error: {e}")

def run_chatbot_conversation():
    try:
        user_chatbot_conversation()
    except Exception as e:
        print(f"Chatbot conversation error: {e}")

def main():
    emotion_detector = EmotionDetector()

    # Create threads for both tasks
    emotion_thread = threading.Thread(target=run_emotion_detection, args=(emotion_detector,))
    chatbot_thread = threading.Thread(target=run_chatbot_conversation)

    # Start both threads
    emotion_thread.start()
    chatbot_thread.start()

    try:
        # Wait for both threads to complete (optional)
        emotion_thread.join()
        chatbot_thread.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
        # Optionally, you could set flags to terminate the threads if they support it
    finally:
        # Clean up if needed
        print("Cleaning up...")

if __name__ == "__main__":
    main()
