import os
import librosa
import numpy as np

DATA_PATH = "data/raw/genres_original"
SAVE_PATH = "data/processed/data.npz"
SAMPLE_RATE = 22050
DURATION = 30 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def prepare_dataset(data_path, n_mels=128, hop_length=512, num_segments=10):
    labels = []
    spectrograms = []
    genres = [g for g in os.listdir(data_path) if not g.startswith('.')]
    
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    for i, genre in enumerate(genres):
        print(f"Processing: {genre}...")
        genre_path = os.path.join(data_path, genre)
        
        for f in os.listdir(genre_path):
            file_path = os.path.join(genre_path, f)
            
            try:
                # Load audio
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Split into segments to augment data
                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    
                    # Extract Mel-Spectrogram
                    melspec = librosa.feature.melspectrogram(
                        y=signal[start:finish], sr=sr, n_mels=n_mels, hop_length=hop_length
                    )
                    melspec_db = librosa.power_to_db(melspec, ref=np.max)
                    
                    # Ensure consistent shape (128 x 130 approx)
                    if melspec_db.shape[1] == 130:
                        spectrograms.append(melspec_db)
                        labels.append(i)
            except Exception as e:
                print(f"Skipping {f}: {e}")

    # Convert to arrays and save
    X = np.array(spectrograms)[..., np.newaxis] # Add channel dimension for CNN
    y = np.array(labels)
    
    np.savez_compressed(SAVE_PATH, X=X, y=y, mapping=np.array(genres))
    print(f"Saved {len(X)} samples to {SAVE_PATH}")

if __name__ == "__main__":
    prepare_dataset(DATA_PATH)