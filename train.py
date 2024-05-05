from collections import namedtuple
from types import SimpleNamespace
import evaluate
import librosa
import numpy as np
from omegaconf import OmegaConf
import pretty_midi
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor, Pop2PianoTokenizer, TrainingArguments, Trainer
import sys
sys.path.append("./pop2piano")

from midi_tokenizer import MidiTokenizer

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
# load the pretrained model, processor, and tokenizer
    model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
    processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
    tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

# load an example audio file and corresponding ground truth midi file
    audio, sr = librosa.load("./processed/audio/Mountain - Mississippi Queen.ogg", sr=44100)  # feel free to change the sr to a suitable value.

    sr = int(sr)

    max_length = 89
    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt", resample=True, max_length=max_length)

    midi = pretty_midi.PrettyMIDI("./processed/midi/Mountain - Mississippi Queen.mid")



    config = OmegaConf.load("./pop2piano/config.yaml")

    # midi_tokenizer = MidiTokenizer(config.tokenizer)
    # beatsteps = np.array(inputs["beatsteps"][0])
    # notes = midi_to_tokens(midi, midi_tokenizer)

    # ARBITRARILY_LARGE = 10000
    # result = midi_tokenizer.relative_batch_tokens_to_midi(notes, beatsteps, cutoff_time_idx=ARBITRARILY_LARGE)
    # result[0].write("test.mid")
    # exit()
    # write("test.mid")
    # print(midi_tokenizer.to_string(notes[0]))
    # print(notes)
    # print(np.shape(np.array(notes) > 140))
    # midi_tokenizer.notes_to_midi(notes, inputs["beatsteps"]).write("test.mid")
    # exit()
    # print(midi.instruments[0].notes[0])
    # notes = np.array([[note.start, note.end, note.pitch, note.velocity] for note in midi.instruments[0].notes])
    # tokens = midi_tokenizer.notes_to_tokens(notes)
    # print(tokens)

    # exit()


    # assuming that labels is longer than max_length
    # labels = np.array(tokenizer.encode_plus(midi.instruments[0].notes, return_tensors="pt")["token_ids"])
    # notes = midi_tokenizer.relative_tokens_to_notes(labels, 0)
    # print(notes)
    # exit()

    # create labels
    labels = np.array(tokenizer(midi.instruments[0].notes, return_tensors="pt", padding="max_length", max_length=max_length)["token_ids"])
    labels = np.array([np.append(labels, np.array([1,0]))]) # pad with EOS token

    # print(inputs["beatsteps"])

    # REALLY IMPORTANT FOR STUPID REASON (tokenizer.num_bars is set to 2 by default)
    tokenizer.num_bars = int(len(inputs["beatsteps"][0]) / 4) # set large number of bars to convert to midi (one batch)

    # print(labels)

    # decode labels according to input beatsteps
    decoded_labels= tokenizer.batch_decode(
            token_ids=labels,
            feature_extractor_output=inputs # contains beatstep information
    )


    # write to midi file
    decoded_labels["pretty_midi_objects"][0].write("decoded.mid")



    # finetune the model
    metric = evaluate.load("accuracy")

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
            )

    print("Fine-tuning model...")
    trainer.train()


    print("Completed fine-tuning.")
