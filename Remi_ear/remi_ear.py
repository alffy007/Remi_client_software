from faster_whisper import WhisperModel
import wave
import pyaudio
import numpy as np
import os
import requests

url = 'http://192.168.1.8:5000/chat'
CYAN = "\033[96m"
RESET_COLOR = "\033[0m"
voice = "Remi_ear/spongebib.mp3"
print("Initializing whisper model with CPU...")
model_size = "tiny.en"
whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("Whisper model initialized successfully.")
def transcribe_with_whisper(audio_file):
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

def detect_silence(data, threshold=1500, chunk_size=1024):
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.mean(np.abs(audio_data)) < threshold

def record_audio(
    file_path, silence_threshold=512, silence_duration=2.0, chunk_size=1024
):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=chunk_size,
    )
    frames = []
    print("Recording...")
    silent_chunks = 0
    speaking_chunks = 0
    while True:
        data = stream.read(chunk_size)
        frames.append(data)
        if detect_silence(data, threshold=silence_threshold, chunk_size=chunk_size) and speaking_chunks==0:
            continue
        elif detect_silence(data, threshold=silence_threshold, chunk_size=chunk_size) and speaking_chunks!=0:    
             silent_chunks += 1
            # print("Silence detected:", silent_chunks)
             if silent_chunks > silence_duration * (16000 / chunk_size):
                break
        else:
            silent_chunks = 0
            speaking_chunks += 1
            # print("Silence detected:", speaking_chunks)
        if speaking_chunks > silence_duration * (16000 / chunk_size) * 10:
            break
    print("Recording stopped.")
    print(speaking_chunks)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b"".join(frames))
    wf.close()

def user_chatbot_conversation():
    try:
        while True:
            audio_file = "temp_recording.wav"
            record_audio(audio_file)
            user_input = transcribe_with_whisper(audio_file)
            os.remove(audio_file)
            data = {
    "chat_text":user_input,
    "mood_prompt":"Happy"
}
            headers = {
                'Content-Type': 'application/json'
            }
            print(CYAN + "You:", user_input +RESET_COLOR)
            response = requests.post(url, json=data, headers=headers)
            print(response.text)

    except KeyboardInterrupt:
        print("\nRemi: Goodbye! Have a great day!")
        exit()                


