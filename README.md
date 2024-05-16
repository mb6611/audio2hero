# Audio2Hero: AI Charting Assistant

#### Audio2Hero is a charting assistant tool that produces chart templates that are actually on-beat! Unlike other autocharters, we use transformers ⚡ instead of predefined algorithms, so the rhythm and button selections are generated in the style of your specific song. You can use the result out of the box, or use it as a starting point to make your own custom charts faster 🎶 !

## HuggingFace Demo
You can use Audio2Hero right now! Check out our demo on Hugging Face: [Demo](https://huggingface.co/spaces/Tim-gubski/Audio2Hero)

## Video Demo
To see Audio2Hero working watch our YouTube Demo: [https://www.youtube.com/watch?v=y23ZDX2WFg0](https://www.youtube.com/watch?v=y23ZDX2WFg0)

## How it works
To learn more about how our model works, check out our paper: [Paper](https://github.com/mb6611/484-clonehero/blob/691cb37b50aafdca9364e9202a797e27dcf9c903/Audio2Hero%20Paper.pdf)

## Known Limitations
- Our pipeline currently does not separate any instruments so the generated chart might actually follow other instruments at parts of the song, like drum solos, baseline, or the vocals. We are looking into ways to fix this but the effect can actually be quite cool at times.
- The model will sometimes overlap sustained notes. This is both an artifact of how our model generates notes and also how our postprocessing works. This will be fixed in a future version
- The generated chart will sometimes exceed the duration of the song audio, so you'll notice some notes appear after the song goes quiet. We believe this is an artifact from our model training and the way we mask padding in the loss function. We may fix this in the future by retraining with a different loss function or add an additional postprocessing step that will trim the chart to the correct length
- Our model only generates medium difficulty at the moment. We started with medium difficulty because we thought it would be an easier starting point to create a proof of concept. We plan to add harder difficulties in the future and overcome the challenges that will come with having a larger volume of notes.


## Questions?
We plan to add better documentation to our GitHub so that others can build off of our work, but in the meantime, if you have any questions about our work feel free to reach out over email!
