import pretty_midi
from matplotlib import pyplot as plt
import librosa

notes = pretty_midi.PrettyMIDI("./clonehero/Aerosmith - Same Old Song & Dance/notes.mid")
notes.instruments[0].notes = [note for note in notes.instruments[0].notes if note.end < 20]

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             colors='hot',
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

plt.figure(figsize=(8, 4))
plt.title("Notes.midi Piano Roll")
plot_piano_roll(notes, 70, 79)
plt.show()