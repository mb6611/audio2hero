from transformers import Pop2PianoForConditionalGeneration

if __name__ == "__main__":
    model = Pop2PianoForConditionalGeneration.from_pretrained("./audio2hero_adafactor_340")
    model.push_to_hub("audio2hero")
