import librosa
import sys
sys.path.append("./pop2piano/evaluate")
from midi_melody_accuracy import evaluate_melody
import numpy as np
import os
import pretty_midi

def dtw_distance(x, y):
    """
    Compute the dynamic time warping distance between two sequences x and y.

    Parameters:
    x, y : numpy arrays
        The sequences to be compared.

    Returns:
    float
        The dynamic time warping distance between the sequences.
    """
    # Lengths of sequences
    m, n = len(x), len(y)

    # Initialize the DTW matrix with zeros
    dtw_matrix = np.zeros((m + 1, n + 1))

    # Fill the first row and first column with infinity
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf

    # Fill the rest of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = abs(x[i - 1] - y[j - 1]) # Cost of matching two elements
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # Insertion
                                           dtw_matrix[i, j - 1],    # Deletion
                                           dtw_matrix[i - 1, j - 1]) # Match

    # Return the DTW distance
    return dtw_matrix[m, n]

if __name__ == "__main__":


    midi_path_1 = "./new_cache/clonehero_processed/audio2hero/AFI - Medicate.mid"
    midi_1 = pretty_midi.PrettyMIDI(midi_path_1)
    audio_1 = midi_1.synthesize()

    # audio_path = "./new_cache/clonehero_processed/audio/All The Young Dudes.ogg"
    audio_path = "./new_cache/clonehero_processed/audio/"
    audio_paths = [os.path.join(audio_path, song) for song in os.listdir(audio_path)][0:5]
    additional_audio_paths = [os.path.join(audio_path, song) for song in os.listdir(audio_path) if "AFI" in song]
    for additional in additional_audio_paths:
        audio_paths.append(additional)
    print(audio_paths)

    for i in range(len(audio_paths)):
        gt_audio_1, sample_rate = librosa.load(audio_paths[i], sr=44100)

        x = librosa.beat.beat_track(y=audio_1, units='time')[1]
        y = librosa.beat.beat_track(y=gt_audio_1, units='time')[1]
        z = dtw_distance(x,y)
        print(f"{audio_paths[i]}: ", z)
#for beat in x[0:20]:
#    plt.axvline(beat, color='r', linestyle='--')
#y = gt_audio_1[2000:600000]
#y = gt_audio_1
#plt.plot(list(range(len(x))), x)
#z = normalized_dot_product(x,y)
    # plt.plot(list(range(len(y))), y)
    # plt.show()


    # files to compare
    # input_midi_path = "frozen_aero.mid"
    # input_audio_path = "./processed/audio/Aerosmith - Same Old Song & Dance.ogg"

    # # evaluate melody
    # frozen_aero_mid = pretty_midi.PrettyMIDI(input_midi_path)
    # gt_audio, _ = librosa.load(input_audio_path, sr=44100)
    # frozen_aero_melody = evaluate_melody(frozen_aero_mid, gt_audio)

    # print("(Raw Chroma Accuracy, Raw Pitch Accuracy)", frozen_aero_melody)
