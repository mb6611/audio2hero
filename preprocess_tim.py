import os, sys, shutil
from tqdm import tqdm
from mergeoggs import merge
from midiprocessor import midi_preprocess

directory = "../GHcharter/Guitar Hero III/Quickplay"

for song in tqdm(os.listdir(directory)):
    if not os.path.isdir(os.path.join(directory, song)):
        continue
    files = os.listdir(os.path.join(directory, song))
    if len(files) == 0:
        continue
    oggs = [os.path.join(directory, song, file) for file in files if file.endswith(".ogg") and file != "preview.ogg"]
    if len(oggs) == 0:
        continue
    # if not os.path.exists(f"./processed/audio/{song}.ogg"):
    # merge(f"./processed/audio/{song}.ogg", *oggs)
    if not os.path.exists(f"./processed/midi_nostrum/{song}.mid"):
        # shutil.copy(os.path.join(directory, song, f"notes.mid"), f"./processed/midi/{song}.mid")
        midi_preprocess(f"{directory}/{song}/notes.mid", f"./processed/midi_nostrum/{song}.mid")
    