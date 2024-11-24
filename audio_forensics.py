import librosa
import numpy as np
from scipy.spatial.distance import cosine

def extract_mfcc(file_path):
    """
    Extract MFCC features from an audio file.
    :param file_path: Path to the audio file
    :return: Mean of MFCC features
    """
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
        # Return mean of MFCC features across time
        mfcc_mean = np.mean(mfcc_features.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def compare_mfcc(mfcc1, mfcc2):
    """
    Compare two MFCC feature arrays using cosine similarity.
    :param mfcc1: MFCC features of first audio file
    :param mfcc2: MFCC features of second audio file
    :return: Cosine similarity score
    """
    try:
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(mfcc1, mfcc2)
        return similarity
    except Exception as e:
        print(f"Error comparing MFCCs: {e}")
        return None

def process_audio_files(suspect_file, call_files):
    """
    Process the suspect's audio file and call recordings for MFCC extraction and matching.
    :param suspect_file: Path to the suspect's audio file
    :param call_files: List of paths to call recordings
    """
    # Extract MFCC features for suspect file
    suspect_mfcc = extract_mfcc(suspect_file)
    if suspect_mfcc is None:
        print("Failed to extract MFCC for suspect file.")
        return

    print(f"MFCC for suspect file ({suspect_file}):\n{suspect_mfcc}\n")

    # Extract MFCC features for each call recording
    for idx, call_file in enumerate(call_files):
        call_mfcc = extract_mfcc(call_file)
        if call_mfcc is None:
            print(f"Failed to extract MFCC for call file {call_file}.")
            continue

        # Compare MFCC features
        similarity = compare_mfcc(suspect_mfcc, call_mfcc)
        if similarity is not None:
            print(f"Cosine Similarity between suspect and call {idx + 1} ({call_file}): {similarity:.2f}")
        else:
            print(f"Failed to compare suspect and call {idx + 1} ({call_file}).")

# Define file paths
suspect_file = 'D:/audio forensis/suspect_sample.wav'  # Path to the suspect's voice sample
call_files = ['D:/audio forensis/call_recordings1.wav','D:/audio forensis/call_recordings2.wav','D:/audio forensis/call_recordings3.wav']  # Paths to call recordings

# Run the audio comparison
process_audio_files(suspect_file, call_files)
