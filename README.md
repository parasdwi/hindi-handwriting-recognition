# Hindi Handwriting Recognition

A Python application for **recognizing one handwritten Hindi (Devanagari) word from a photograph**. It uses TensorFlow/Keras for model inference, OpenCV and Pillow for image preprocessing, and Gradio for a browser-based interface.

**Repository:** https://github.com/parasdwi/hindi-handwriting-recognition

## Features

- Upload a photograph of one handwritten Hindi word, or capture one using a camera supported by the browser.
- Optionally auto-crop the image around the writing.
- Optionally correct uneven lighting and shadows.
- See the recognized text alongside the cropped/contrast-adjusted image and the 32 × 128 model input.

> **Scope:** The current interface is designed for *one handwritten word per image*, not full-page text recognition or document OCR.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio web interface and app entry point |
| `recognizer.py` | Image preprocessing, model loading and CTC decoding |
| `model_def.py` | CNN, Spatial Transformer, BiLSTM and self-attention model definition |
| `charset.json` | Output-character vocabulary |
| `v4_best.weights.h5` | Trained model weights; stored with Git LFS |
| `requirements.txt` | Python dependencies |

## Requirements

- Python (Python 3.10 or 3.11 is a reasonable starting point; verify against your known-working environment)
- Git and **Git LFS** (to fetch the trained weights)
- A working internet connection for initial dependency installation and cloning

## Installation and execution (Windows PowerShell)

1. Install Git and Git LFS, then run:

   ```powershell
   git lfs install
   git clone https://github.com/parasdwi/hindi-handwriting-recognition.git
   cd hindi-handwriting-recognition
   git lfs pull
   ```

2. Create a Python virtual environment:

   ```powershell
   py -3.10 -m venv .venv
   ```

   If Python 3.10 is not installed, use another compatible installed Python version and adjust the command.

3. Install dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Start the application:

   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

5. Open the local address printed in the terminal (typically `http://127.0.0.1:7860`). Upload a photo of one handwritten Hindi word and click **Recognize**.

## How it works

1. **Image input:** Gradio supplies a photo to the recognition pipeline.
2. **Preprocessing:** Correct EXIF orientation, handle transparent backgrounds, convert to grayscale, and optionally correct lighting and crop to the ink region.
3. **Normalization:** Stretch contrast; resize to height 32 while preserving aspect ratio; invert and pad to a 32 × 128 array; scale pixel values to `[0, 1]`.
4. **Feature extraction:** A convolutional model with a Spatial Transformer extracts features from the normalized word image.
5. **Sequence recognition:** Bidirectional LSTM layers and self-attention produce a sequence of character-class probabilities.
6. **Decoding:** Greedy CTC-style decoding removes repeated time steps and blank tokens to produce Devanagari characters.
7. **Output:** The recognized word and two preprocessing visualizations appear in the app.

```text
Handwritten Hindi word photo
          |
          v
Grayscale + optional crop / lighting correction
          |
          v
Contrast adjustment + resize / invert / pad (32 x 128)
          |
          v
CNN + Spatial Transformer + BiLSTM + Self-Attention
          |
          v
Character probabilities -> greedy CTC decoding
          |
          v
Recognized Hindi word
```

## Reported performance and limitations

The application's interface currently reports approximately **92% character accuracy** and **78% exact-word accuracy** on word crops from phone photographs by 12 unseen writers. It also warns that exact-word accuracy on out-of-vocabulary words was around half. These are **figures stated in the existing app**, not independently verified by this README. Results vary with handwriting, lighting, image framing and vocabulary.

The public repository currently contains the inference application and model weights. **Training scripts and training data are not included** in the listed files; add them (or document their availability) if a reviewer needs to reproduce training rather than only run inference.

## Troubleshooting

- **Weights cannot be loaded:** Confirm that Git LFS is installed and run `git lfs pull`. The `.h5` file should be a real model file, not a tiny Git LFS pointer text file.
- **A package cannot be installed:** Check your Python version and use the dependency versions from the environment in which the project was successfully tested.
- **Poor recognition:** Try a tightly cropped, well-lit image containing just one handwritten word; toggle the crop and lighting settings.

## Submission

This public GitHub repository provides the implementation. The reviewer can clone it and follow the installation steps above. To establish reproducibility, run the documented commands from a fresh environment before submission, and pin dependency versions known to work together.
