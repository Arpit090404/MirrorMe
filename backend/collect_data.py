import cv2
import sounddevice as sd
import numpy as np
import os
from datetime import datetime

def record_session(duration=180, fps=30, output_dir='../data/recordings'):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_out = cv2.VideoWriter(f'{output_dir}/{timestamp}_video.mp4', fourcc, fps, (640, 480))
    
    audio_data = []
    def audio_callback(indata, frames, time, status):
        audio_data.append(indata.copy())
    
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
    stream.start()
    
    print("Recording... Press 'q' to stop early.")
    start_time = cv2.getTickCount()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > duration:
            break
    
    stream.stop()
    audio_np = np.concatenate(audio_data, axis=0)
    np.save(f'{output_dir}/{timestamp}_audio.npy', audio_np)
    
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
    print(f"Saved: {timestamp}_video.mp4 and _audio.npy")

if __name__ == "__main__":
    record_session()