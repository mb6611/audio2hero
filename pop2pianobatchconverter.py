import librosa
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor
import os, sys
from tqdm import tqdm
import torch

# feel free to change the sr to a suitable value.
directory = "./clonehero/audio"


start = 0
count = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
model_hero = Pop2PianoForConditionalGeneration.from_pretrained("./models/audio2hero_adam2_150").to(device)
model_pop = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano").to(device)
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

for file in tqdm(os.listdir(directory)):
  try:
    print(f"{directory}/{file}")
    audio, sr = librosa.load(f"{directory}/{file}" , sr=44100)  
  except:
    print(f"Error processing {file}")
    continue

  inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
  model_output_hero = model_hero.generate(
      input_features=inputs["input_features"],
  )
  model_output_pop = model_pop.generate(
      input_features=inputs["input_features"],
  )
  tokenizer_output_hero = processor.batch_decode(
      token_ids=model_output_hero, feature_extractor_output=inputs
  )["pretty_midi_objects"][0]
  tokenizer_output_pop = processor.batch_decode(
      token_ids=model_output_pop, feature_extractor_output=inputs
  )["pretty_midi_objects"][0]

  # Since we now have 2 generated MIDI files
  out_name = file.split(".")[0]+".mid"
  tokenizer_output_hero.write(f"./clonehero/audio2hero/{out_name}")
  tokenizer_output_pop.write(f"./clonehero/pop2piano/{out_name}")
  break