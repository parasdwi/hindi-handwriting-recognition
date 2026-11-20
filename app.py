import gradio as gr
from PIL import Image

from recognizer import Recognizer

rec = Recognizer()

CSS = "#word textarea {font-size: 2.2rem; line-height: 1.5;}"


def recognize(img, crop, fix_light):
    if img is None:
        return "", None, None
    text, gray, x = rec.predict(img, crop=crop, fix_light=fix_light)
    return (text if text else "(nothing recognised)"), Image.fromarray(gray), Image.fromarray(x)


with gr.Blocks(title="Hindi handwriting recognizer", css=CSS) as demo:
    gr.Markdown(
        "# Hindi handwriting recognizer\n"
        "Take a photo of **one handwritten Hindi word** (dark pen, light paper, even light) "
        "or upload one, then press Recognize. On a phone, the upload button offers the camera."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", sources=["upload", "webcam"],
                           label="Photo of one handwritten word")
            crop = gr.Checkbox(True, label="Auto-crop around the word")
            light = gr.Checkbox(False, label="Fix uneven lighting / shadows")
            btn = gr.Button("Recognize", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Recognised word", lines=1, elem_id="word")
            with gr.Row():
                cropped = gr.Image(label="After crop and contrast", interactive=False)
                seen = gr.Image(label="What the model sees (32x128)", interactive=False)
    gr.Markdown(
        "Accuracy note: on word images cropped from phone photos by 12 writers the model never saw, "
        "about 92% of characters were right and about 78% of whole words were exactly right. "
        "For words outside its training vocabulary, only about half were exactly right. "
        "Your own handwriting can do better or worse. If the result is wrong, try a tighter photo "
        "or toggle the options. Photos are processed in memory and not saved by this app."
    )
    btn.click(recognize, [inp, crop, light], [out, cropped, seen])

if __name__ == "__main__":
    demo.launch()
