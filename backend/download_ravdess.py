import urllib.request
import zipfile
import os

# Create data dir
os.makedirs('../data/ravdess/audio', exist_ok=True)
os.makedirs('../data/ravdess/video', exist_ok=True)

# Download audio speech (~215 MB)
print("Downloading audio...")
urllib.request.urlretrieve("https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1", "ravdess_audio.zip")
with zipfile.ZipFile("ravdess_audio.zip", 'r') as zip_ref:
    zip_ref.extractall('../data/ravdess/audio')
os.remove("ravdess_audio.zip")

# Download video for Actor 01 (~553 MB)
print("Downloading video for Actor 01...")
urllib.request.urlretrieve("https://zenodo.org/records/1188976/files/Video_Speech_Actor_01.zip?download=1", "ravdess_video_actor01.zip")
with zipfile.ZipFile("ravdess_video_actor01.zip", 'r') as zip_ref:
    zip_ref.extractall('../data/ravdess/video')
os.remove("ravdess_video_actor01.zip")

print("Dataset downloaded and extracted to data/ravdess. Ready for processing.")