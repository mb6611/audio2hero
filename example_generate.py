import librosa
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor, Pop2PianoTokenizer, Pop2PianoConfig
import pretty_midi
from transformers import AutoConfig

# import matplotlib.pyplot as plt
# import mido
# import numpy as np
# import torch
# from torch import nn
# from torch.nn import CrossEntropyLoss


if __name__ == "__main__":

    # load the pretrained model, processor, and tokenizer
    # config = AutoConfig.from_pretrained("sweetcocoa/pop2piano")
    config = Pop2PianoConfig.from_pretrained("sweetcocoa/pop2piano")
    og_model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
    generation_config = og_model.generation_config
    model = Pop2PianoForConditionalGeneration._from_config(config)
    processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
    tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

    # load an example audio file and corresponding ground truth midi file
    # audio, sr = librosa.load("./processed/audio/Mountain - Mississippi Queen.ogg", sr=44100)  # feel free to change the sr to a suitable value.
    audio, sr = librosa.load("./processed/audio/Aerosmith - Same Old Song & Dance.ogg", sr=44100)  # feel free to change the sr to a suitable value.

    # inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt", resample=False)
    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")

    # midi = pretty_midi.PrettyMIDI("./processed/midi/Mountain - Mississippi Queen.mid")
    # labels = tokenizer.encode_plus(midi.instruments[0].notes, return_tensors="pt")

    # generate model output
    print("Generating output...")
    model_output = model.generate(inputs["input_features"], generation_config=generation_config, output_logits=True, return_dict_in_generate=True)
    print("Completed generation.")

    # decode model output
    print("Decoding output...")
    tokenizer_output = processor.batch_decode(
            token_ids=model_output.sequences,
            feature_extractor_output=inputs
        )

    tokenizer_output["pretty_midi_objects"][0].write("output.mid")
    print(tokenizer_output.keys())
