import librosa
import sys
sys.path.append("./pop2piano/evaluate")
import math
# from midi_melody_accuracy import evaluate_melody
import numpy as np
import os
import pickle
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

def compute_dtw(audio_path, midi_path, output):

    # audio_path = "./new_cache/clonehero_processed/audio/All The Young Dudes.ogg"
    audio_paths = os.listdir(audio_path)
    # additional_audio_paths = [os.path.join(audio_path, song) for song in os.listdir(audio_path) if "AFI" in song]
    # for additional in additional_audio_paths:
    #     audio_paths.append(additional)
    # print(audio_paths)

    audio_paths = audio_paths[1:10]

    result = []
    for song1 in audio_paths:
        song1_result = []
        try:

            print(os.path.join(midi_path, song1.replace(".ogg", ".mid")))
            midi_1 = pretty_midi.PrettyMIDI(os.path.join(midi_path, song1.replace(".ogg", ".mid")))
            audio_1 = midi_1.synthesize()

            for song2 in audio_paths:
                # gt_audio_1, sample_rate = librosa.load(audio_paths[i], sr=44100)
                try:
                    gt_audio_1, sample_rate = librosa.load(os.path.join(audio_path, song2), sr=44100)

                    x = librosa.beat.beat_track(y=audio_1, units='time')[1]
                    y = librosa.beat.beat_track(y=gt_audio_1, units='time')[1]
                    z = dtw_distance(x,y)
                    song1_result.append(z)
                    print(f"{(song1, song2)}: ", z)
                except Exception as e:
                    song1_result.append(-1)
                    print("Failed to process song: ", song2)
            result_np = np.array(song1_result)
            result.append(result_np / np.linalg.norm(result_np))
        except Exception as e:
            print("Failed to process song: ", song1)
            result.append(np.pad(song1_result, (0, len(audio_paths) - len(song1_result)), 'constant', constant_values=-1))
    pickle.dump(result, open(f"{output}", "wb"))

def compute_lambda(similarity_function, audio_path, midi_path, output):

    audio_paths = os.listdir(audio_path)
    # audio_paths = audio_paths[6:10]
    # audio_paths = audio_paths[1:25]
    audio_paths = audio_paths[20:40]

    audio_paths = [song for song in audio_paths if os.path.exists(os.path.join(midi_path, song.replace(".ogg", ".mid")))]

    result = []
    for song1 in audio_paths:
        song1_result = []
        try:

            print(os.path.join(midi_path, song1.replace(".ogg", ".mid")))
            midi_1 = pretty_midi.PrettyMIDI(os.path.join(midi_path, song1.replace(".ogg", ".mid")))
            audio_1 = midi_1.synthesize()

            for song2 in audio_paths:
                # gt_audio_1, sample_rate = librosa.load(audio_paths[i], sr=44100)
                try:
                    gt_audio_1, _ = librosa.load(os.path.join(audio_path, song2), sr=44100)

                    x = librosa.beat.beat_track(y=audio_1, units='time')[1]
                    y = librosa.beat.beat_track(y=gt_audio_1, units='time')[1]
                    # pad the shorter sequence with zeros
                    if len(x) > len(y):
                        y = np.pad(y, (0, len(x) - len(y)), 'constant', constant_values=0)
                    if len(y) > len(x):
                        x = np.pad(x, (0, len(y) - len(x)), 'constant', constant_values=0)
                    z = similarity_function(x, y)
                    if math.isnan(z):
                        raise Exception("nan")
                    song1_result.append(z)
                    print(f"{(song1, song2)}: ", z)
                except Exception as e:
                    print(e)
                    song1_result.append(-1)
                    print("Failed to process song: ", song2)
            result_np = np.array(song1_result)
            result.append(result_np / np.linalg.norm(result_np))
        except Exception as e:
            print("Failed to process song: ", song1)
            result.append(np.pad(song1_result, (0, len(audio_paths) - len(song1_result)), 'constant', constant_values=-1))
    pickle.dump(result, open(f"{output}", "wb"))




import os
import math
import pickle
import pretty_midi
import librosa
import numpy as np
from multiprocessing import Pool

def parallel_process_song(song1, audio_paths, midi_path, similarity_function):
    song1_result = []
    try:
        midi_1 = pretty_midi.PrettyMIDI(os.path.join(midi_path, song1.replace(".ogg", ".mid")))
        audio_1 = midi_1.synthesize()

        for song2 in audio_paths:
            try:
                gt_audio_1, _ = librosa.load(os.path.join(audio_path, song2), sr=44100)

                x = librosa.beat.beat_track(y=audio_1, units='time')[1]
                y = librosa.beat.beat_track(y=gt_audio_1, units='time')[1]

                if len(x) > len(y):
                    y = np.pad(y, (0, len(x) - len(y)), 'constant', constant_values=0)
                if len(y) > len(x):
                    x = np.pad(x, (0, len(y) - len(x)), 'constant', constant_values=0)
                z = similarity_function(x, y)
                if math.isnan(z):
                    raise Exception("nan")
                song1_result.append(z)
            except Exception as e:
                song1_result.append(-1)
                print("Failed to process song: ", song2)
        return song1_result
    except Exception as e:
        print("Failed to process song: ", song1)
        return np.pad(song1_result, (0, len(audio_paths) - len(song1_result)), 'constant', constant_values=-1)

def parallel_compute_lambda(similarity_function, audio_path, midi_path, output):
    audio_paths = os.listdir(audio_path)
    audio_paths = audio_paths[1:25]

    result = []
    with Pool() as pool:
        for song1_result in pool.imap_unordered(
            lambda song1: parallel_process_song(song1, audio_paths, midi_path, similarity_function),
            audio_paths
        ):
            result.append(song1_result)

    result_np = np.array(result)
    return result_np
    # pickle.dump(result_np, open(f"{output}", "wb"))
    # result_normalized = result_np / np.linalg.norm(result_np)
    # pickle.dump(result_normalized, open(f"{output}_normalized", "wb"))























def normalized_dot_product(a, b):
    """
    Compute the normalized dot product between two sequences a and b.

    Parameters:
    a, b : numpy arrays
        The sequences to be compared.

    Returns:
    float
        The normalized dot product between the sequences.
    """
    # Compute the dot product
    dot_product = np.dot(a, b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return -1

    # Normalize the dot product
    return dot_product / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluate_chroma_accuracy(song: str):

    """

    Inputs:

    song: str - the name of the song to evaluate formatted as "*.ogg", "*.mp3", etc.

    """


    audio2hero_path = "./new_cache/clonehero_processed/audio2hero/"
    pop2piano_path = "./new_cache/clonehero_processed/pop2piano/"
    audio_path = os.path.join("./new_cache/clonehero_processed/audio/", song)
    # input_audio_path = "./processed/audio/Aerosmith - Same Old Song & Dance.ogg"
    # input_audio_path = "./processed/audio/Aerosmith - Same Old Song & Dance.ogg"

    song_midi = song.replace(".ogg", ".mid")

    # # evaluate melody
    audio2hero_midi = pretty_midi.PrettyMIDI(os.path.join(audio2hero_path, song_midi))
    gt_audio, _ = librosa.load(audio_path, sr=44100)
    audio2hero_accuracies = evaluate_melody(audio2hero_midi, gt_audio)

    print("audio2hero: (Raw Chroma Accuracy, Raw Pitch Accuracy)", audio2hero_accuracies)

    # # evaluate melody
    pop2piano_midi = pretty_midi.PrettyMIDI(os.path.join(pop2piano_path, song_midi))
    gt_audio, _ = librosa.load(audio_path, sr=44100)
    pop2piano_accuracies = evaluate_melody(pop2piano_midi, gt_audio)

    print("pop2piano: (Raw Chroma Accuracy, Raw Pitch Accuracy)", pop2piano_accuracies)

def generate_audio_pickles(audio_path, output_path):

    audio_paths = os.listdir(audio_path)

    for song in audio_paths:
        try:
            audio, _ = librosa.load(os.path.join(audio_path, song), sr=44100)
            pickle.dump(audio, open(os.path.join(output_path, song.replace(".ogg", ".pkl")), "wb"))
        except Exception as e:
            print("Failed to process song: ", song)

if __name__ == "__main__":


    # midi_path_1 = "./new_cache/clonehero_processed/audio2hero/AFI - Medicate.mid"
    # midi_1 = pretty_midi.PrettyMIDI(midi_path_1)
    # audio_1 = midi_1.synthesize()
    # midi_path = "./new_cache/clonehero_processed/midi/"
    midi_path = "./new_cache/clonehero_processed/audio2hero/"

    # audio_path = "./new_cache/clonehero_processed/audio/All The Young Dudes.ogg"
    audio_path = "./new_cache/clonehero_processed/audio/"
    audio_paths = os.listdir(audio_path)

    # evaluate_chroma_accuracy("AFI - Medicate.ogg")

    # compute_dtw(audio_path, midi_path, "dtw_distance.pkl")
    # compute_lambda(np.corrcoef, audio_path, midi_path, "corrcoef.pkl")
    compute_lambda(normalized_dot_product, audio_path, midi_path, "dot_big_audio2hero2.pkl")

    # result = parallel_compute_lambda(normalized_dot_product, audio_path, midi_path, "dot_big.pkl")
    # pickle.dump(result, open("dot_big.pkl", "wb"))
    # compute_lambda(np.cov, audio_path, midi_path, "cov.pkl")






    # additional_audio_paths = [os.path.join(audio_path, song) for song in os.listdir(audio_path) if "AFI" in song]
    # for additional in additional_audio_paths:
    #     audio_paths.append(additional)
    # print(audio_paths)

    # audio_paths = audio_paths[1:10]

    # result = []
    # for song1 in audio_paths:
    #     song1_result = []
    #     try:

    #         print(os.path.join(midi_path, song1.replace(".ogg", ".mid")))
    #         midi_1 = pretty_midi.PrettyMIDI(os.path.join(midi_path, song1.replace(".ogg", ".mid")))
    #         audio_1 = midi_1.synthesize()

    #         for song2 in audio_paths:
    #             # gt_audio_1, sample_rate = librosa.load(audio_paths[i], sr=44100)
    #             try:
    #                 gt_audio_1, sample_rate = librosa.load(os.path.join(audio_path, song2), sr=44100)

    #                 x = librosa.beat.beat_track(y=audio_1, units='time')[1]
    #                 y = librosa.beat.beat_track(y=gt_audio_1, units='time')[1]
    #                 z = dtw_distance(x,y)
    #                 song1_result.append(z)
    #                 print(f"{(song1, song2)}: ", z)
    #             except Exception as e:
    #                 song1_result.append(-1)
    #                 print("Failed to process song: ", song2)
    #         result_np = np.array(song1_result)
    #         result.append(result_np / np.linalg.norm(result_np))
    #     except Exception as e:
    #         print("Failed to process song: ", song1)
    #         result.append(np.pad(song1_result, (0, len(audio_paths) - len(song1_result)), 'constant', constant_values=-1))
    # pickle.dump(result, open("dtw_distance.pkl", "wb"))

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

