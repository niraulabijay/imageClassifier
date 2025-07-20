import gradio as gr
from fastai.vision.all import *


def is_cat(x): return x[0].isUpper()

learn = load_learner('carOrNotModel.pkl')

categories = ('dog','cat')
def classify_images(img):
  pred, idx, probs = learn.predict(img)
  return dict(zip(categories,map(float,probs)))

image = gr.Image(height=192, width=192)
label = gr.Label()
examples = ['dog.png','cat.jpeg','dunno.jpg']

intf = gr.Interface(fn=classify_images, inputs = image, outputs=label, examples=examples)

# code that starts the ui with url:
intf.launch(inline=False, share=true)


