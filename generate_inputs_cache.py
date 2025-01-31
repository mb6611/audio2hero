import evaluate
import librosa
from midi_loss_calculator import MIDILossCalculator, one_hot_convert, preprocess_labels, pad_labels
import multiprocessing
import numpy as np
import os
import pickle
import pretty_midi
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor, Pop2PianoTokenizer, TrainingArguments, Trainer
import sys
sys.path.append("./pop2piano")

# import matplotlib.pyplot as plt
# import mido
# import numpy as np
import torch
# from torch import nn
# from torch.nn import CrossEntropyLoss

def midi_to_tokens(midi_obj, tokenizer):
        """
        Converts a pretty_midi object to tokens using the provided tokenizer.

        Args:
            midi_obj (pretty_midi.PrettyMIDI): A pretty_midi object.
            tokenizer (MidiTokenizer): The tokenizer object to use.

        Returns:
            np.ndarray: An array of tokens representing the MIDI data.
        """
        notes = []
        for note in midi.instruments[0].notes:
            onset = int(note.start * midi_obj.resolution // 24)  # Convert to time step index
            offset = int(note.end * midi_obj.resolution // 24)  # Convert to time step index
            pitch = note.pitch
            velocity = note.velocity
            notes.append([onset, offset, pitch, velocity])
        notes = np.array(notes)
        # notes = []
        # for note in midi.instruments[0].notes:
        #     onset_idx = np.searchsorted(tokenizer.config.beatstep, note.start)
        #     offset_idx = np.searchsorted(tokenizer.config.beatstep, note.end)
        #     pitch = note.pitch
        #     velocity = note.velocity
        #     notes.append([onset_idx, offset_idx, pitch, velocity])
        return notes


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



def process_song(args):
  song_name, processor, tokenizer = args
  audio_path = f"{audio_dir}{song_name}.ogg"
  ground_truth_midi_path = f"{ground_truth_midi_dir}{song_name}.mid"
  if not os.path.exists(audio_path) or not os.path.exists(ground_truth_midi_path):
    # continue
    return
#   print(f"Audio file: {audio_path}")
#   print(f"Ground truth midi file: {ground_truth_midi_path}")
  try:


    #   print("Loading audio file...")

    if os.path.exists(f"{cache_dir}{song_name}.pkl"):
        inputs, labels, gt_longest_length = pickle.load(open(f"{cache_dir}{song_name}.pkl", "rb"))
        print(f"{song_name} already cached.")
        #   print("Loaded from cache.")
        # labels, gt_longest_length = np.load("./cache/preprocessed_labels/Aerosmith - Same Old Song & Dance.npy", allow_pickle=True)

    else:
        print(f"Processing {song_name}")
        # audio_path = "./processed/audio/Pat Benatar - Hit Me with Your Best Shot.ogg"
        audio, sr = librosa.load(audio_path, sr=44100)  # feel free to change the sr to a suitable value.
        # audio, sr = librosa.load(audio_path, sr=22050)  # feel free to change the sr to a suitable value.
        #   print("Loaded audio file.\n")

        sr = int(sr)

        # convert the audio file to tokens
        # inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt", resample=True)
        inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
        #   inputs = {k: v.to(device) for k, v in inputs.items()}

        # load ground truth midi file
        # midi = pretty_midi.PrettyMIDI("./processed/midi/Mountain - Mississippi Queen.mid")
        # ground_truth_midi_path = "mountain_out_gen.mid"
        #   print("Encoding ground truth midi file...")
        # ground_truth_midi_path = "./processed/audio/Aerosmith - Same Old Song and Dance.ogg"
        midi = pretty_midi.PrettyMIDI(ground_truth_midi_path)


        labels, gt_longest_length = preprocess_labels(midi, inputs, tokenizer)
        pickle.dump((inputs, labels, gt_longest_length), open(f"{cache_dir}{song_name}.pkl", "wb"))
        print(f"Cached {song_name}.")
        # np.save("Aerosmith - Same Old Song & Dance.npy", np.array([labels, gt_longest_length]))
  except:

    print(f"Error in {song_name}")
    return
    # continue








if __name__ == "__main__":

    MidiLossCalculator = MIDILossCalculator()

    # load the pretrained model, processor, and tokenizer
    # model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
    # processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
    # tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

    print("Loading pretrained model, processor, and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = Pop2PianoForConditionalGeneration.from_pretrained("./cache/model").to(device)
    processor = Pop2PianoProcessor.from_pretrained("./cache/processor")
    tokenizer = Pop2PianoTokenizer.from_pretrained("./cache/tokenizer")


    print("Loaded pretrained model, processor, and tokenizer.\n")
    # cache the model, processor, and tokenizer to avoid downloading them again
    # model.save_pretrained("./cache/model")
    # processor.save_pretrained("./cache/processor")
    # tokenizer.save_pretrained("./cache/tokenizer")

    # load an example audio file and corresponding ground truth midi file
    # audio_path = "./processed/audio/Mountain - Mississippi Queen.ogg"

    model.train()
    lr=1e-4
    momentum=0
    for param in model.parameters():
        param.requires_grad_(True) # or False
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # print number of parameters
    print(f"Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # audio_dir = "./processed/audio/"
    audio_dir = "/home/dataset_storage/clonehero_processed/audio/"
    # ground_truth_midi_dir = "./processed/midi/"
    ground_truth_midi_dir = "/home/dataset_storage/clonehero_processed/midi/"
    # cache_dir = "./cache/preprocessed_labels/"
    cache_dir = "/home/dataset_storage/clonehero_processed/preprocessed_labels/"

    song_names = os.listdir(audio_dir)
    song_names = [".".join(song_name.split(".")[0:-1]) for song_name in song_names]
    # print(song_names)

    # losses = []
    # for epoch in range(400):
    #   print(f"Epoch {epoch+1}")
    #   avg_loss = 0
    #   epoch_losses = []
      # for song_name in song_names:

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_song, [(song_name, processor, tokenizer) for song_name in song_names])



            # generate model output
            #   print("Generating output...")
            # inputs = {k: v.to(device) for k, v in inputs.items()}
            # model_output = model.generate(inputs["input_features"], generation_config=model.generation_config, return_dict_in_generate=True, output_logits=True, min_new_tokens=gt_longest_length)
            #   print("Completed generation.\n")


            # longest_length = len(model_output.sequences[0])
            # padded_labels = pad_labels(labels, longest_length)


            # print(f"Labels shape:", padded_labels.shape)

            #   print("Encoded ground truth midi file.\n")


            # decode model output
            #   print("Decoding output...")
            #   tokenizer_output = processor.batch_decode(
            #           token_ids=model_output.sequences,
            #           feature_extractor_output=inputs
            #       )

            # logits = torch.stack(model_output.logits).transpose(0,1).requires_grad_()
            # t_labels = torch.tensor(padded_labels).to(device)
            # t_labels = t_labels[:,1:]
            # one_hot = one_hot_convert(t_labels, 2400)
            # one_hot = one_hot.to(device)

            # optimizer.zero_grad()

            # midi_loss = MidiLossCalculator.cross_entropy_loss(logits, one_hot)
            # midi_loss.backward()
            # optimizer.step()

            # avg_loss += midi_loss.item()
            # epoch_losses.append(midi_loss.item())
            # print("Loss:", midi_loss.item())
          # except:
          #   print(f"Error in {song_name}")
          #   continue
      # losses.append(epoch_losses)
      # np.save("losses.npy", np.array(losses))
      # if (epoch+1) % 5 == 0:
      #   model.save_pretrained(f"./models/audio2hero_{epoch+1}")
      # print("Average loss:", avg_loss/len(epoch_losses))
