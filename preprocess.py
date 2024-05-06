import os, sys, shutil
from tqdm import tqdm
from mergeoggs import merge
from midiprocessor import midi_preprocess

# directory = "./Guitar Hero III/Quickplay"
# directory = "/home/dataset_storage/clonehero/Band Hero"
directory = "/home/dataset_storage/clonehero/Bonus"
processed_dir = "/home/dataset_storage/clonehero_processed"

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
    # print(os.path.join(processed_dir, "audio", f"{song}.ogg"))
    # merge(f"./processed/audio/{song}.ogg", *oggs)
    song_midi_path = os.path.join(processed_dir, "midi", f"{song}.mid")
    merge(os.path.join(processed_dir, "audio", f"{song}.ogg"), *oggs)
    if not os.path.exists(f"./processed/midi/{song}.mid"):
        shutil.copy(os.path.join(directory, song, f"notes.mid"), song_midi_path)
        midi_preprocess(midi_in=f"{directory}/{song}/notes.mid", midi_out=song_midi_path)
    # if not os.path.exists(f"./processed/midi/{song}.mid"):
    #     shutil.copy(os.path.join(directory, song, f"notes.mid"), f"./processed/midi/{song}.mid")
    #     midi_preprocess(midi_in=f"{directory}/{song}/notes.mid", midi_out=f"./processed/midi/{song}.mid")
