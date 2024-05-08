import py_midicsv as pm
import pandas as pd
import mido
import os


def midi_preprocess(midi_in, midi_out):
  csv_string = pm.midi_to_csv(midi_in)
  df = pd.DataFrame([x.split(", ") for x in csv_string])
  # notes = ['78','77','76','75','74','73','72','71', None]
  notes = ['75','74','73','72','71', None] # get rid of strum
  #get rid of notes not in list
  df = df[df[4].isin(notes) | ~(df[2].isin(['Note_on_c', 'Note_off_c']))]
  csv = df.values.tolist()
  csv = [', '.join([i for i in x if i != None]) for x in csv]

  # Parse the CSV output of the previous command back into a MIDI file
  midi_object = pm.csv_to_midi(csv)

  # Save the parsed MIDI file to disk
  with open(midi_in + "_temp.mid", "wb") as output_file:
      midi_writer = pm.FileWriter(output_file)
      midi_writer.write(midi_object)

  song = mido.MidiFile(midi_in + '_temp.mid')
  #remove all tracks without name "PART GUITAR"
  for i, track in enumerate(song.tracks.copy()):
      if "PART GUITAR" not in track.name:
          song.tracks.remove(track)

  song.save(midi_out)

  os.remove(midi_in + '_temp.mid')


if __name__ == "__main__":
    midi_preprocess("notes.mid", "notes_processed.mid")
