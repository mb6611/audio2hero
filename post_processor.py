import pretty_midi
from matplotlib import pyplot as plt
import librosa
import copy
import mido
import configparser
import os
import shutil
import random

output_dir = "./clonehero"

# for file in os.listdir("./processed/piano_midi"):
# song_name = ".".join(file.split(".")[0:-1])
song_name = "Dire Straits - Sultans of Swing"
# song_name = "Aerosmith - Same Old Song & Dance"
print(song_name)

if not os.path.exists(f"{output_dir}/{song_name}"):
    os.makedirs(f"{output_dir}/{song_name}")

# copy over song
# shutil.copy(f"./processed/audio/{song_name}.ogg", f"{output_dir}/{song_name}/song.ogg")
shutil.copy(f"./{song_name}.ogg", f"{output_dir}/{song_name}/song.ogg")

# notes = pretty_midi.PrettyMIDI(f"./processed/piano_midi/{file}")
notes = pretty_midi.PrettyMIDI(f"./sultans_ada.mid")

output = copy.deepcopy(notes)
output.instruments = []
output.instruments.append(pretty_midi.Instrument(0, name="PART GUITAR"))
last_start = 0

note_times = [note.start for note in notes.instruments[0].notes if note.pitch != 78]

total = 0
outofrange = 0
for index,note in enumerate(notes.instruments[0].notes):
    time_start = note.start
    # if time_start == last_start:
    #     continue
    # if index % 2 != 0:
    #     continue
    total+=1
    if note.pitch not in [71,72,73,74,75, 78]:
        outofrange+=1
    last_start = time_start
    # new_pitch = 71 + note.pitch % 5
    new_pitch = note.pitch 
    duration = note.end - note.start
    end = note.start + duration if duration > 0.5 else note.start + 0.1
    new_note = pretty_midi.Note(velocity=100, pitch=new_pitch, start=note.start, end=end)

    # if strum
    if note.pitch == 78 and note.start not in note_times:
        extra_note = pretty_midi.Note(velocity=100, pitch=random.randint(71,75), start=note.start, end=end)
        output.instruments[0].notes.append(extra_note)
    # strum = pretty_midi.Note(velocity=100, pitch=78, start=note.start, end=end)
    output.instruments[0].notes.append(new_note)
    # output.instruments[0].notes.append(strum)

print(f"Total notes: {total}")
print(f"Out of range notes: {outofrange}")

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