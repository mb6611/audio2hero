import gradio as gr
from transformers import pipeline
from audio2hero import generate_midi
import spaces



gradio_app = gr.Interface(
    spaces.GPU(generate_midi),
    inputs=gr.Audio(label="Input Audio", type="filepath"),
    outputs=gr.File(label="Output MIDI Zip File"),
    title="Audio2Hero AI Charter Assistant for CloneHero",
    description="""Audio2Hero will generate a medium difficulty Clone Hero chart from any audio file. This can be a helping starting point to create your own charts, or you can play them as is! 
                  Make sure to rename your audio file to 'Artist - Song Name' for the autocharter to correctly
                  generate the song.ini file. The output will be a zip file containing the MIDI file, song.ogg 
                  and song.ini file. The auto charter can take upto 45 seconds to generate so please be patient. Hope you enjoy!""",
)

if __name__ == "__main__":
    gradio_app.launch()