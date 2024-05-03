import pretty_midi
from matplotlib import pyplot as plt
import librosa
import copy
import mido
import configparser
import os
import shutil

output_dir = "./clonehero"

for file in os.listdir("./processed/piano_midi"):
    song_name = ".".join(file.split(".")[0:-1])
    print(song_name)

    if not os.path.exists(f"{output_dir}/{song_name}"):
        os.makedirs(f"{output_dir}/{song_name}")

    # copy over song
    shutil.copy(f"./processed/audio/{song_name}.ogg", f"{output_dir}/{song_name}/song.ogg")

    notes = pretty_midi.PrettyMIDI(f"./processed/piano_midi/{file}")

    output = copy.deepcopy(notes)
    output.instruments = []
    output.instruments.append(pretty_midi.Instrument(0, name="PART GUITAR"))
    last_start = 0
    for index,note in enumerate(notes.instruments[0].notes):
        time_start = note.start
        if time_start == last_start:
            continue
        if index % 2 != 0:
            continue
        last_start = time_start
        new_pitch = 71 + note.pitch % 5
        duration = note.end - note.start
        end = note.start + duration if duration > 0.5 else note.start + 0.1
        new_note = pretty_midi.Note(velocity=100, pitch=new_pitch, start=note.start, end=end)
        strum = pretty_midi.Note(velocity=100, pitch=78, start=note.start, end=end)
        output.instruments[0].notes.append(new_note)
        output.instruments[0].notes.append(strum)

    output.write(f"{output_dir}/{song_name}/notes.mid")

    output = mido.MidiFile(f"{output_dir}/{song_name}/notes.mid")
    output.tracks[1].pop(1)
    output.save(f"{output_dir}/{song_name}/notes.mid")

    # write ini file
    config = configparser.ConfigParser()
    config.read('./song.ini')
    config.set("song", "name", song_name.split(" - ")[1])
    config.set("song", "artist", song_name.split(" - ")[0])
    config.set("song", "charter", "Tim and Matthew")

    with open(f"{output_dir}/{song_name}/song.ini", 'w') as configfile:    # save
        config.write(configfile)