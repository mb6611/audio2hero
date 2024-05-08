import os, sys, shutil
from tqdm import tqdm
from mergeoggs import merge
from midiprocessor import midi_preprocess
import multiprocessing


def process_dir(song):
    # directory, processed_dir = args
    try:
        if not os.path.isdir(os.path.join(directory, song)):
            return
            # continue
        files = os.listdir(os.path.join(directory, song))
        if len(files) == 0:
            return
            # continue
        oggs = [os.path.join(directory, song, file) for file in files if file.endswith(".ogg") and file != "preview.ogg"]
        if len(oggs) == 0:
            return
            # continue
        # if not os.path.exists(f"./processed/audio/{song}.ogg"):
        # print(os.path.join(processed_dir, "audio", f"{song}.ogg"))
        # merge(f"./processed/audio/{song}.ogg", *oggs)
        processed_midi_path = os.path.join(processed_dir, "midi", f"{song}.mid")
        merge(os.path.join(processed_dir, "audio", f"{song}.ogg"), *oggs)
        if not os.path.exists(processed_midi_path):
            shutil.copy(os.path.join(directory, song, f"notes.mid"), processed_midi_path)
            # midi_preprocess(midi_in=f"{processed_dir}/midi/{song}.mid", midi_out=song_midi_path)
            midi_preprocess(midi_in=processed_midi_path, midi_out=processed_midi_path)
            print("Successfully preprocessed: ", song)
        # if not os.path.exists(f"./processed/midi/{song}.mid"):
        #     shutil.copy(os.path.join(directory, song, f"notes.mid"), f"./processed/midi/{song}.mid")
        #     midi_preprocess(midi_in=f"{directory}/{song}/notes.mid", midi_out=f"./processed/midi/{song}.mid")
        print("Successfully processed: ", song)
        return
    except Exception as e:
        print(e)
        print(f"Error processing {song}")
        return
        # continue

if __name__ == "__main__":

    # params = directory, processed_dir
    # directory = "./Guitar Hero III/Quickplay"
    # directory = "/home/dataset_storage/clonehero/Band Hero"
    # directory = "/home/dataset_storage/clonehero/Bonus"
    # directory = "/home/dataset_storage/clonehero/DJ Hero Guitar Charts"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero/Bonus"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero/Quickplay"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero - Metallica"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero - Smash Hits" # BAD FORMATTING
    # directory = "/home/dataset_storage/clonehero/Guitar Hero - Warriors of Rock" # BAD
    # directory = "/home/dataset_storage/clonehero/Guitar Hero 5" # MOSTLY DONE?
    # directory = "/home/dataset_storage/clonehero/Guitar Hero 5 Hidden Songs" # SUBFOLDERS
    # directory = "/home/dataset_storage/clonehero/Guitar Hero Encore Rocks the 80s"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero Encore Rocks the 80s Tutorials"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero II" # BUG?
    # directory = "/home/dataset_storage/clonehero/Guitar Hero II Tutorials"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero On Tour"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero Tutorials"
    # directory = "/home/dataset_storage/clonehero/Guitar Hero Van Halen" # CRASHED?
    # directory = "/home/dataset_storage/clonehero/Guitar Hero World Tour"
    # directory = "/home/dataset_storage/clonehero/Quickplay"
    # processed_dir = "/home/dataset_storage/clonehero_processed"
    directory = "../GHcharter/Guitar Hero III/Quickplay"
    processed_dir = "./processed/midi_nostrum"

    # process_dir(os.listdir(directory)[1])
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tqdm(pool.map(process_dir, os.listdir(directory)))

    exit()


# for song in tqdm(os.listdir(directory)):
#     try:
#         if not os.path.isdir(os.path.join(directory, song)):
#             continue
#         files = os.listdir(os.path.join(directory, song))
#         if len(files) == 0:
#             continue
#         oggs = [os.path.join(directory, song, file) for file in files if file.endswith(".ogg") and file != "preview.ogg"]
#         if len(oggs) == 0:
#             continue
#         # if not os.path.exists(f"./processed/audio/{song}.ogg"):
#         # print(os.path.join(processed_dir, "audio", f"{song}.ogg"))
#         # merge(f"./processed/audio/{song}.ogg", *oggs)
#         song_midi_path = os.path.join(processed_dir, "midi", f"{song}.mid")
#         merge(os.path.join(processed_dir, "audio", f"{song}.ogg"), *oggs)
#         if not os.path.exists(f"./processed/midi/{song}.mid"):
#             shutil.copy(os.path.join(directory, song, f"notes.mid"), song_midi_path)
#             midi_preprocess(midi_in=f"{directory}/{song}/notes.mid", midi_out=song_midi_path)
#         # if not os.path.exists(f"./processed/midi/{song}.mid"):
#         #     shutil.copy(os.path.join(directory, song, f"notes.mid"), f"./processed/midi/{song}.mid")
#         #     midi_preprocess(midi_in=f"{directory}/{song}/notes.mid", midi_out=f"./processed/midi/{song}.mid")
#     except:
#         print(f"Error processing {song}")
#         continue
