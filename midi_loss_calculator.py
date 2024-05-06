from encoder import encode_plus
import librosa
import numpy as np
import pretty_midi
import torch
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor, Pop2PianoTokenizer
from typing import List

from decode import crop_midi

def one_hot_convert(t_labels, vocab_size):
    # Your vocabulary size
    vocab_size = 2400

    # Create a tensor to hold the one-hot encoded versions
    one_hot_tensor = torch.zeros((*t_labels.shape, vocab_size))

    # Iterate over each element of the original tensor
    for i in range(t_labels.size(0)):
        for j in range(t_labels.size(1)):
            # Get the value from the original tensor
            value = int(t_labels[i, j])
            # One-hot encode the value
            one_hot = torch.zeros(vocab_size)
            one_hot[value] = 1
            # Assign it to the corresponding position in the new tensor
            one_hot_tensor[i, j] = one_hot
    return one_hot_tensor


class MIDILossCalculator:

    # def __init(self):
    #     pass

    def cross_entropy_loss(
            self,
            generated_midi_logits: torch.Tensor,
            ground_truth_midi_tokens: torch.Tensor,
        ):

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        output = cross_entropy_loss(generated_midi_logits, ground_truth_midi_tokens)

        return output

    def note_density_loss(
            self,
            generated_midi_tokens,
            ground_truth_midi_tokens
        ):
        pass

def preprocess_labels(midi):
    batches = [crop_midi(midi, i, i+8, inputs.extrapolated_beatstep[0]).instruments[0].notes for i in range(2, len(inputs.extrapolated_beatstep[0])-10, 8)]

    labels = []
    for batch in batches:
        label, _ = encode_plus(tokenizer, batch, return_tensors="pt", time_offset=0)
        labels.append(label["token_ids"])
    labels = [np.append([0], np.append(label, [1, 0])) for label in labels]
    gt_longest_length = max([len(label) for label in labels])

    return labels, gt_longest_length

def pad_labels(labels, longest_model_output):
    padded_labels = np.array([np.pad(label, (0, longest_model_output - len(label))) for label in labels])
    return padded_labels

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
    ground_truth_midi_path = "./processed/piano_midi/Pat Benatar - Hit Me with Your Best Shot.mid"
    midi = pretty_midi.PrettyMIDI(ground_truth_midi_path)




    # convert the midi file to tokens
    # batches = [crop_midi(midi, i, i+8, inputs.extrapolated_beatstep[0]).instruments[0].notes for i in range(2, len(inputs.extrapolated_beatstep[0])-10, 8)]

    # # format labels as (batch_size, tokens_per_batch)
    # labels = []
    # for batch in batches:
    #     label, _ = encode_plus(tokenizer, batch, return_tensors="pt", time_offset=0)
    #     labels.append(label["token_ids"])
    # labels = [np.append([0], np.append(label, [1, 0])) for label in labels]

    # gt_longest_length = max([len(label) for label in labels])

    labels, gt_longest_length = preprocess_labels(midi)


    model_output = model.generate(inputs["input_features"], generation_config=model.generation_config, return_dict_in_generate=True, output_logits=True, min_new_tokens=gt_longest_length)


    longest_length = len(model_output.sequences[0])
    padded_labels = pad_labels(labels, longest_length)


    # 
    # print(longest_length)
    # padded_labels = np.array([np.pad(label, (0, longest_length - len(label))) for label in labels])
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
    print(np.array(padded_labels).shape)
    print(len(model_output["logits"]))
    print(np.array(model_output["logits"]).shape)
    logits = torch.Tensor(np.array(model_output["logits"]))

    # format logits to (
    # logits = logits.reshape(logits.shape[1], logits.shape[0], logits.shape[2])
    logits = torch.stack(model_output.logits).transpose(0,1)
    t_labels = torch.tensor(padded_labels)
    t_labels = t_labels[:,1:]
    print("One hotting labels...")
    one_hot = one_hot_convert(t_labels, 2400)
    print("One hotted labels.\n")

    print(logits.shape)
    print(one_hot.shape)

    midi_loss = MidiLossCalculator.cross_entropy_loss(logits, one_hot)
    print(midi_loss)




    # print(model_output["logits"][2].shape)

    # tokenizer_output["pretty_midi_objects"][0].write("output.mid")
    # print(tokenizer_output.keys())
