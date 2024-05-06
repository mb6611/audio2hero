from encoder import encode_plus
import librosa
import numpy as np
import pretty_midi
import torch
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor, Pop2PianoTokenizer
from typing import List

from decode import crop_midi

class MIDILossCalculator:

    # def __init(self):
    #     pass

    def cross_entropy_loss(
            self,
            generated_midi_logits: torch.Tensor,
            ground_truth_midi_tokens: torch.Tensor,
        ) -> torch.nn.CrossEntropyLoss:

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        output = cross_entropy_loss(generated_midi_logits, ground_truth_midi_tokens)

        return output

    def note_density_loss(
            self,
            generated_midi_tokens,
            ground_truth_midi_tokens
        ):
        pass

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

    print("Loading audio file...")
    audio_path = "./processed/audio/Pat Benatar - Hit Me with Your Best Shot.ogg"
    # audio, sr = librosa.load(audio_path, sr=44100)  # feel free to change the sr to a suitable value.
    audio, sr = librosa.load(audio_path, sr=22050)  # feel free to change the sr to a suitable value.
    print("Loaded audio file.\n")

    sr = int(sr)

    # convert the audio file to tokens
    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt", resample=True)


    # load ground truth midi file
    # midi = pretty_midi.PrettyMIDI("./processed/midi/Mountain - Mississippi Queen.mid")
    # ground_truth_midi_path = "mountain_out_gen.mid"
    print("Encoding ground truth midi file...")
    ground_truth_midi_path = "./processed/piano_midi/Pat Benatar - Hit Me with Your Best Shot.mid"
    midi = pretty_midi.PrettyMIDI(ground_truth_midi_path)

    # convert the midi file to tokens
    batches = [crop_midi(midi, i, i+8, inputs.extrapolated_beatstep[0]).instruments[0].notes for i in range(2, len(inputs.extrapolated_beatstep[0])-10, 8)]

    # format labels as (batch_size, tokens_per_batch)
    labels = []
    offset = 0
    for batch in batches:
        print(f"outer offset: {offset}")
        label, offset = encode_plus(tokenizer, batch, return_tensors="pt", time_offset=0)
        labels.append(label["token_ids"])

    labels = [np.append([0], np.append(label, [1, 0])) for label in labels]
    longest_length = max([len(label) for label in labels])
    # padded_labels = np.array([np.pad(label, (0, longest_length - len(label))) for label in labels])
    padded_labels = torch.Tensor(np.array([np.pad(label, (0, longest_length - len(label))) for label in labels]))
    # print(padded_labels[2])



    # labels = [tokenizer(batch, return_tensors="pt")['token_ids'] for batch in batches]
    # labels = [np.append([0], np.append(label, [1, 0])) for label in labels]
    # longest_length = max([len(label) for label in labels])
    # padded_labels = np.array([np.pad(label, (0, longest_length - len(label))) for label in labels])

    # padded_labels[padded_labels > 135] = 135
    print(f"Labels shape:", padded_labels.shape)

    print("Encoded ground truth midi file.\n")

    # generate model output
    print("Generating output...")
    model_output = model.generate(inputs["input_features"], output_logits=True, return_dict_in_generate=True)
    print("Completed generation.\n")

    # decode model output
    print("Decoding output...")
    tokenizer_output = processor.batch_decode(
            token_ids=model_output.sequences,
            feature_extractor_output=inputs
        )

    print(model_output.sequences.shape)
    print(np.array(labels["token_ids"]).shape)
    print(len(model_output["logits"]))
    print(np.array(model_output["logits"]).shape)
    logits = torch.Tensor(np.array(model_output["logits"]))

    # format logits to (
    # logits = logits.reshape(logits.shape[1], logits.shape[0], logits.shape[2])

    midi_loss = MidiLossCalculator.cross_entropy_loss(logits, padded_labels)
    print(midi_loss)



    # print(model_output["logits"][2].shape)

    # tokenizer_output["pretty_midi_objects"][0].write("output.mid")
    # print(tokenizer_output.keys())
