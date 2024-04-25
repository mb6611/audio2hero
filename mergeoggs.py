from pydub import AudioSegment


def merge(out_dir, *files):
    if len(files)==0:
        raise ValueError("No files to merge")
    combined = AudioSegment.from_file(files[0], format="ogg")
    for file in files[1:]:
        sound = AudioSegment.from_file(file, format="ogg")
        combined = combined.overlay(sound)
    combined.export(out_dir, format="ogg")

if __name__ == "__main__":
    merge("merged.ogg","guitar.ogg", "bass.ogg", "song.ogg")