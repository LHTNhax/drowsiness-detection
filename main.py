import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import gradio as gr
from tf_keras.models import load_model

from preprocessing import preprocess_for_inference

MODEL_PATH  = "model.keras"
CLASS_NAMES = ["drowsy", "notdrowsy"]
CLASS_EMOJI = {"drowsy": "😴", "notdrowsy": "😊"}

print(f"Loading model from '{MODEL_PATH}' …")
model = load_model(MODEL_PATH)
print("Model ready.")

def predict_image(img_rgb: np.ndarray):
    if img_rgb is None:
        return "_No image provided._", None

    model_input, annotated, face_found = preprocess_for_inference(img_rgb)

    probs = model.predict(model_input, verbose=0)[0]   
    idx   = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    conf  = float(probs[idx])

    emoji = CLASS_EMOJI[label]

    bar_filled = int(conf * 20)
    bar        = "█" * bar_filled + "░" * (20 - bar_filled)
    result_md  = (
        f"## {emoji} {label.upper()}\n\n"
        f"**Confidence:** {conf:.1%}\n\n"
        f"`{bar}` {conf:.1%}\n\n"
    )

    if not face_found:
        result_md = (
            "> ⚠️ **No face detected** — prediction may be unreliable.\n\n"
            + result_md
        )

    result_md += "### Class probabilities\n"
    for i, name in enumerate(CLASS_NAMES):
        result_md += f"- **{name}**: {probs[i]:.2%}\n"

    return result_md, annotated

DESCRIPTION = """
# 😴 Drowsiness Detection

Upload an image **or** use your webcam to check for drowsiness.

The model automatically detects your **face**, **eyes**, and **mouth** before
making a prediction.

| Box colour | Region |
|-----------|--------|
| 🟩 Green  | Face   |
| 🟧 Orange | Eyes   |
| 🟦 Blue   | Mouth  |
"""

with gr.Blocks(
    title="Drowsiness Detection",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    ),
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.Tab("📁 Upload Image"):
            with gr.Row():
                with gr.Column():
                    upload_input = gr.Image(
                        label="Input Image",
                        type="numpy",
                        image_mode="RGB",
                    )
                    upload_btn = gr.Button(
                        "🔍 Predict", variant="primary", size="lg")

                with gr.Column():
                    upload_annotated = gr.Image(
                        label="Detected Regions",
                        type="numpy",
                        interactive=False,
                    )
                    upload_result = gr.Markdown(label="Result")

            upload_btn.click(
                fn=predict_image,
                inputs=upload_input,
                outputs=[upload_result, upload_annotated],
            )

            upload_input.change(
                fn=predict_image,
                inputs=upload_input,
                outputs=[upload_result, upload_annotated],
            )

        with gr.Tab("📷 Webcam"):
            gr.Markdown(
                "Allow camera access when the browser asks, "
                "then click **Capture & Predict**."
            )

            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(
                        label="Webcam",
                        sources=["webcam"],
                        type="numpy",
                        image_mode="RGB",
                    )
                    webcam_btn = gr.Button(
                        "📸 Capture & Predict", variant="primary", size="lg")

                with gr.Column():
                    webcam_annotated = gr.Image(
                        label="Detected Regions",
                        type="numpy",
                        interactive=False,
                    )
                    webcam_result = gr.Markdown(label="Result")

            webcam_btn.click(
                fn=predict_image,
                inputs=webcam_input,
                outputs=[webcam_result, webcam_annotated],
            )

    gr.Markdown(
        "---\n"
        "*Model: InceptionV3 fine-tuned on Drowsiness Detection dataset.*"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   
        server_port=7860,        
        share=False,
        show_error=True,
    )