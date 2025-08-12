# audio_utils.py
import numpy as np
import librosa
import soundfile as sf

def load_audio_file(path, sr=22050):
    """
    Load audio from disk using librosa; returns (y, sr)
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def frame_features(y, sr=22050, frame_len_s=1.0):
    """
    Split audio into contiguous frames (frame_len_s seconds) and compute simple features:
    returns list of dicts {'rms', 'centroid', 'key'}
    """
    hop = int(sr * frame_len_s)
    frames = []
    for i in range(0, len(y), hop):
        seg = y[i:i+hop]
        if len(seg) == 0:
            break
        if len(seg) < hop:
            seg = np.pad(seg, (0, hop - len(seg)))
        rms = float(np.mean(seg**2) + 1e-12)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key = int(np.argmax(chroma_mean))
        frames.append({'rms': rms, 'centroid': centroid, 'key': key})
    return frames

def compute_obs_from_frames(frame_a, frame_b, vol_a, vol_b):
    tempo_diff = 0.0
    energy_diff = abs(frame_a['rms'] - frame_b['rms'])
    centroid_diff = abs(frame_a['centroid'] - frame_b['centroid']) / 5000.0
    diff = min((frame_a['key'] - frame_b['key']) % 12, (frame_b['key'] - frame_a['key']) % 12)
    key_compat = max(0.0, 1.0 - (diff / 6.0))
    return np.array([tempo_diff, energy_diff, centroid_diff, key_compat, vol_a, vol_b], dtype=np.float32)

def render_mix(y_a, y_b, vols_a, vols_b, sr=22050):
    """
    Apply per-frame volumes (1s frames) and overlap-add to produce mixed output.
    """
    frame_hop = sr  # 1 second frames
    n_frames = max(len(vols_a), len(vols_b))
    max_len = max(len(y_a), len(y_b), n_frames * frame_hop)
    out = np.zeros(max_len + sr, dtype=np.float32)
    for i in range(n_frames):
        start = i * frame_hop
        end = start + frame_hop
        seg_a = y_a[start:end] if start < len(y_a) else np.zeros(frame_hop)
        seg_b = y_b[start:end] if start < len(y_b) else np.zeros(frame_hop)
        if len(seg_a) < frame_hop: seg_a = np.pad(seg_a, (0, frame_hop - len(seg_a)))
        if len(seg_b) < frame_hop: seg_b = np.pad(seg_b, (0, frame_hop - len(seg_b)))
        mix_seg = seg_a * vols_a[i] + seg_b * vols_b[i]
        out[start:start+frame_hop] += mix_seg
    # normalize
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.99:
        out = out / peak * 0.98
    return out
