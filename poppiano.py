import librosa
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

audio, sr = librosa.load("./processed/audio/Aerosmith - Same Old Song & Dance.ogg", sr=44100)  # feel free to change the sr to a suitable value.
model = Pop2PianoForConditionalGeneration.from_pretrained("./models/audio2hero_230")
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
model_output = model.generate(input_features=inputs["input_features"])
tokenizer_output = processor.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)["pretty_midi_objects"][0]
tokenizer_output.write("./audio2hero_output.mid")