import io
import librosa
import numpy as np
import soundfile as sf

def extract_features_from_audio(audio, sr):

    audio = audio.astype(np.float32)

    features = []

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    zcr = librosa.feature.zero_crossing_rate(audio)

    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    return np.array(features, dtype=np.float32)


def extract_features(audio_dict):

    audio_bytes = audio_dict["bytes"]

    audio, sr = sf.read(
        io.BytesIO(audio_bytes)
    )

    return extract_features_from_audio(audio, sr)

def extract_features_from_file(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    return extract_features_from_audio(audio, sr)