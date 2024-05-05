import evaluate
import librosa
import numpy as np
import pretty_midi
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor, Pop2PianoTokenizer, TrainingArguments, Trainer

# import matplotlib.pyplot as plt
# import mido
# import numpy as np
import torch
# from torch import nn
# from torch.nn import CrossEntropyLoss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# load the pretrained model, processor, and tokenizer
model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

# load an example audio file and corresponding ground truth midi file
audio, sr = librosa.load("./processed/audio/Mountain - Mississippi Queen.ogg", sr=44100)  # feel free to change the sr to a suitable value.

sr = int(sr)
inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt", resample=False)

midi = pretty_midi.PrettyMIDI("./processed/midi/Mountain - Mississippi Queen.mid")
labels = tokenizer.encode_plus(midi.instruments[0].notes, return_tensors="pt")

# print(labels)
# labels = torch.tensor(labels["token_ids"]).reshape(83, -1)
print(len(labels["token_ids"]))
exit()

decoded_labels= tokenizer.batch_decode(
        token_ids=labels,
        feature_extractor_output=inputs
)

exit()

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


# model.train()
print("Completed fine-tuning.")



# generate model output
# print("Generating output...")
# model_output = model.generate(inputs["input_features"], output_logits=True, return_dict_in_generate=True)
# print("Completed generation.")

# decode model output
# print("Decoding output...")
# tokenizer_output = processor.batch_decode(
#         token_ids=model_output.sequences,
#         feature_extractor_output=inputs
#     )

# print(tokenizer_output.keys())
