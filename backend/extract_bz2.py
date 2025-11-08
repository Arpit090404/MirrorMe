import bz2
with bz2.BZ2File('shape_predictor_68_face_landmarks.dat.bz2', 'rb') as f:
    with open('shape_predictor_68_face_landmarks.dat', 'wb') as out:
        out.write(f.read())
print("Extracted successfully.")