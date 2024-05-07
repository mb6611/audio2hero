import librosa
import sys
sys.path.append("./pop2piano/evaluate")
from midi_melody_accuracy import evaluate_melody
import pretty_midi


if __name__ == "__main__":

    # files to compare
    input_midi_path = "frozen_aero.mid"
    input_audio_path = "./processed/audio/Aerosmith - Same Old Song & Dance.ogg"

    # evaluate melody
    frozen_aero_mid = pretty_midi.PrettyMIDI(input_midi_path)
    gt_audio, _ = librosa.load(input_audio_path, sr=44100)
    frozen_aero_melody = evaluate_melody(frozen_aero_mid, gt_audio)

    print("(Raw Chroma Accuracy, Raw Pitch Accuracy)", frozen_aero_melody)
