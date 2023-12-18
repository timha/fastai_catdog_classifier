import gradio as gr
from fastai.vision.all import *
import skimage


def is_cat(x): return x[0].isupper()
learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def predict(img):
    img = PILImage.create(img)

    # Resize image if necessary
    img = img.resize((192, 192))
    pred,pred_idx,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


title = "Cat or Dog classifier"
description = "A Cat or Dog classifier trained on the _ dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p></p>"
examples = ['Grey-cat.jpeg']


gr.Interface(fn=predict,inputs=gr.Image(),outputs="label",title=title,description=description,article=article,examples=examples).launch(share=True)
