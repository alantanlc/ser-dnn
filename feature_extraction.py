#   FEATURE EXTRACTION
#
#   We are going to collect five different features from each audio and fuse them in a vector. These are:
#
#   melspectrogram:     Compute a Mel-scaled power spectrogram
#   mfcc:               Mel-frequency cepstral coefficients
#   chroma-stft:        Compute a chromagram from a waveform or power spectrogram
#   spectral_contrast:  Compute a spectral contrast
#   tonnetz:            Computes the tonal centroid features (tonnetz)
#
#   Each audio file will have a fixed vector size of 193.
#
#   The two functions defined below will collect both the features and emotion labels from the audio files.

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
                print("Error encountered while parsing file: ", fn)
                continue
            ext_feature = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_feature])
            labels = np.append(labels, fn.split('\\')[2].split('-')[2])
    return np.array(features), np.array(labels, dtype = np.int)
