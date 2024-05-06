from collections import namedtuple
from types import SimpleNamespace
import evaluate
import librosa
from midi_loss_calculator import MIDILossCalculator, one_hot_convert, preprocess_labels, pad_labels
import numpy as np
import os
from omegaconf import OmegaConf
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


if __name__ == "__main__":

    MidiLossCalculator = MIDILossCalculator()

    # load the pretrained model, processor, and tokenizer
    # model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
    # processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
    # tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

    print("Loading pretrained model, processor, and tokenizer...")
    model = Pop2PianoForConditionalGeneration.from_pretrained("./cache/model")
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
    lr=1e-3
    momentum=0.9
    for param in model.parameters():
        param.requires_grad_(True) # or False
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    audio_path = "./processed/audio/Aerosmith - Same Old Song & Dance.ogg"
    ground_truth_midi_path = "./processed/piano_midi/Aerosmith - Same Old Song & Dance.mid"
    audio_path_data = [(audio_path, ground_truth_midi_path) for i in range(4)]

    for audio_path, ground_truth_midi_path in audio_path_data:
        print(f"Audio file: {audio_path}")
        print(f"Ground truth midi file: {ground_truth_midi_path}")



        print("Loading audio file...")

        if os.path.exists("./cache/preprocessed_labels/Aerosmith - Same Old Song & Dance.npy"):
            labels, gt_longest_length = pickle.load(open("./cache/preprocessed_labels/Aerosmith - Same Old Song & Dance.pkl", "rb"))
            print("Loaded from cache.")
            # labels, gt_longest_length = np.load("./cache/preprocessed_labels/Aerosmith - Same Old Song & Dance.npy", allow_pickle=True)

        else:
            # audio_path = "./processed/audio/Pat Benatar - Hit Me with Your Best Shot.ogg"
            audio, sr = librosa.load(audio_path, sr=44100)  # feel free to change the sr to a suitable value.
            # audio, sr = librosa.load(audio_path, sr=22050)  # feel free to change the sr to a suitable value.
            print("Loaded audio file.\n")

            sr = int(sr)

            # convert the audio file to tokens
            # inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt", resample=True)
            inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")


            # load ground truth midi file
            # midi = pretty_midi.PrettyMIDI("./processed/midi/Mountain - Mississippi Queen.mid")
            # ground_truth_midi_path = "mountain_out_gen.mid"
            print("Encoding ground truth midi file...")
            # ground_truth_midi_path = "./processed/audio/Aerosmith - Same Old Song and Dance.ogg"
            midi = pretty_midi.PrettyMIDI(ground_truth_midi_path)


            labels, gt_longest_length = preprocess_labels(midi, inputs, tokenizer)
            pickle.dump((labels, gt_longest_length), open("./cache/preprocessed_labels/Aerosmith - Same Old Song & Dance.pkl", "wb"))
            # np.save("Aerosmith - Same Old Song & Dance.npy", np.array([labels, gt_longest_length]))


        # generate model output
        print("Generating output...")
        model_output = model.generate(inputs["input_features"], generation_config=model.generation_config, return_dict_in_generate=True, output_logits=True, min_new_tokens=gt_longest_length)
        print("Completed generation.\n")


        longest_length = len(model_output.sequences[0])
        padded_labels = pad_labels(labels, longest_length)


        # print(f"Labels shape:", padded_labels.shape)

        print("Encoded ground truth midi file.\n")


        # decode model output
        print("Decoding output...")
        tokenizer_output = processor.batch_decode(
                token_ids=model_output.sequences,
                feature_extractor_output=inputs
            )

        logits = torch.stack(model_output.logits).transpose(0,1).requires_grad_()
        t_labels = torch.tensor(padded_labels)
        t_labels = t_labels[:,1:]
        one_hot = one_hot_convert(t_labels, 2400)

        optimizer.zero_grad()

        midi_loss = MidiLossCalculator.cross_entropy_loss(logits, one_hot)
        midi_loss.backward()
        optimizer.step()

        print("Loss:", midi_loss.item())
