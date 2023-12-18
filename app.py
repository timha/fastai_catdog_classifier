import gradio as gr
from fastai.vision.all import *
import skimage


def is_cat(x): return x[0].isupper()
learn = load_learner('model.pkl')

labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)

    # Resize image if necessary
    img = img.resize((512, 512))
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p></p>"
examples = ['Grey-cat.jpeg']


gr.Interface(fn=predict,inputs=gr.Image(),outputs=gr.Label(),title=title,description=description,article=article,examples=examples).launch(share=True)
