import copy
import librosa
import numpy as np
import pretty_midi
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor, Pop2PianoTokenizer
from encoder import encode_plus
import sys
sys.path.append("./pop2piano")




# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


import copy
def crop_midi(midi, start_beat, end_beat, extrapolated_beatsteps):
    start = extrapolated_beatsteps[start_beat]
    end = extrapolated_beatsteps[end_beat]
    out = copy.deepcopy(midi)
    for note in out.instruments[0].notes.copy():
        if note.start > end or note.start < start:
            out.instruments[0].notes.remove(note)
        # interpolate index of start note

        lower = np.argmax(extrapolated_beatsteps[extrapolated_beatsteps <= note.start])
        note.start = lower
        note.start = int(note.start - start_beat)

        lower = np.argmax(extrapolated_beatsteps[extrapolated_beatsteps <= note.end])
        note.end = lower
        note.end = int(note.end - start_beat)
        if note.end == note.start:
            note.end += 1
    return out

if __name__ == "__main__":

    # load the pretrained model, processor, and tokenizer
    # model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
    # processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
    # tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

    model = Pop2PianoForConditionalGeneration.from_pretrained("./cache/model")
    processor = Pop2PianoProcessor.from_pretrained("./cache/processor")
    tokenizer = Pop2PianoTokenizer.from_pretrained("./cache/tokenizer")

    print("Loaded pretrained model, processor, and tokenizer.")
    # cache the model, processor, and tokenizer to avoid downloading them again
    # model.save_pretrained("./cache/model")
    # processor.save_pretrained("./cache/processor")
    # tokenizer.save_pretrained("./cache/tokenizer")

    # load an example audio file and corresponding ground truth midi file
    # audio_path = "./processed/audio/Mountain - Mississippi Queen.ogg"
    audio_path = "./processed/audio/Pat Benatar - Hit Me with Your Best Shot.ogg"
    audio, sr = librosa.load(audio_path, sr=44100)  # feel free to change the sr to a suitable value.

    sr = int(sr)

    # convert the audio file to tokens
    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt", resample=True)


    # load ground truth midi file
    # midi = pretty_midi.PrettyMIDI("./processed/midi/Mountain - Mississippi Queen.mid")
    # ground_truth_midi_path = "./processed/midi/Mountain - Mississippi Queen.mid"
    # ground_truth_midi_path = "mountain_out_gen.mid"
    ground_truth_midi_path = "./processed/piano_midi/Pat Benatar - Hit Me with Your Best Shot.mid"
    midi = pretty_midi.PrettyMIDI(ground_truth_midi_path)


    # # convert the midi file to tokens
    batches = [crop_midi(midi, i, i+8, inputs.extrapolated_beatstep[0]).instruments[0].notes for i in range(2, len(inputs.extrapolated_beatstep[0])-8, 8)]
    print(batches[2])
    # # remove empty batches
    # batches = [batch for batch in batches if len(batch) > 0]

    labels = []
    offset = 0
    for batch in batches:
        print(f"outer offset: {offset}")
        label, offset = encode_plus(tokenizer, batch, return_tensors="pt", time_offset=0)        
        labels.append(label["token_ids"])
    labels = [np.append([0], np.append(label, [1, 0])) for label in labels]
    longest_length = max([len(label) for label in labels])
    padded_labels = np.array([np.pad(label, (0, longest_length - len(label))) for label in labels])
    print(padded_labels[2])

    # padded_labels[padded_labels > 135] = 135


    # # decode the tokens
    tokenizer.num_bars = 2
    output = tokenizer.batch_decode(np.array(padded_labels),feature_extractor_output=inputs)

    # # write the decoded midi file
    output_file_path = "output.mid"
    output['pretty_midi_objects'][0].write(output_file_path)
