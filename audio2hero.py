import sys
import librosa
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor, Pop2PianoTokenizer, Pop2PianoConfig
import pretty_midi
from transformers import AutoConfig
from model_generate import generate
import torch
from post_processor import post_process
import tempfile



if __name__=="__main__":
  args = sys.argv[1:]
  song_path = args[0]
  output_dir = args[1]

  print("Loading Model...")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = Pop2PianoForConditionalGeneration.from_pretrained("./models/audio2hero_adafactor_340").to(device)
  model.eval()
  processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
  tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

  print("Processing Song...")
  # load an example audio file and corresponding ground truth midi file
  audio, sr = librosa.load(song_path, sr=44100)  # feel free to change the sr to a suitable value.
  inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")


  # generate model output
  print("Generating output...")
  model.generation_config.output_logits = True
  model.generation_config.return_dict_in_generate = True
  model_output = model.generate(inputs["input_features"].to(device))

  tokenizer_output = processor.batch_decode(
          token_ids=model_output.sequences.cpu(),
          feature_extractor_output=inputs
      )

  # save to temp file
  temp_dir = tempfile.TemporaryDirectory()
  tokenizer_output["pretty_midi_objects"][0].write(f"{temp_dir.name}/temp.mid")

  print("Post Processing...")
  post_process(song_path, f"{temp_dir.name}/temp.mid", output_dir)

  temp_dir.cleanup()
  print("Done.")
