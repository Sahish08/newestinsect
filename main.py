#importing required libraries
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import cv2
import pandas as pd
from google.cloud import storage

app = Flask(__name__)

#insect classification model
model=torchvision.models.regnet_y_32gf()
weights=torch.load('model.pth',map_location=torch.device('cpu'))
model.fc=torch.nn.Linear(3712,142)
model.load_state_dict(weights,strict=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model.eval()
#contains names of insects
cmnDf = pd.read_csv('insectNames_new.csv')

#takes the inputted image and transforms it into a tensor for the PyTorch model
def transforms_validation(image):
    crop_size = 224
    resize_size = 256
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = InterpolationMode.BILINEAR
    transforms_val = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std)])
    image = Image.fromarray(np.uint8(image))
    image = transforms_val(image).reshape((1, 3, 224, 224))
    return image

#runs the PyTorch model on the inputted image
def evaluate(model,image,cmnDf):
    model.eval()
    device=torch.device('cpu')
    image=transforms_validation(image)
    file=open('classes.txt','r')
    classes=[]
    content=file.readlines()
    for i in content:
        spl=i.split('\n')[0]
        classes.append(spl)

    with torch.inference_mode():
        image = image.to(device, non_blocking=True)
        output = model(image)
        op = torch.nn.functional.softmax(output)
        op_ix = torch.argmax(op)
        sciPred = classes[op_ix]
        cmnPred = cmnDf.loc[cmnDf['Scientific name'] == sciPred, 'Common Name'].iloc[0]
        confirmed = False
        if(op[0][op_ix] >= 0.97):
            confirmed = True
        return sciPred, cmnPred, confirmed

#calls the previous helper functions and returns the predictions 
def get_prediction(PATH_TO_IMAGE):
    image = cv2.imread(PATH_TO_IMAGE)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    sciPred,cmnPred,confirmed = evaluate(model, image, cmnDf)
    return sciPred, cmnPred,confirmed
#uploads the image to the google cloud storage bucket
def upload_image(buck,source,destination):
    myClient = storage.Client()
    dest_bucket = myClient.bucket(buck)
    img = dest_bucket.blob(destination)
    img.upload_from_filename(source)

# routes

#connects to frontend
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

#about page if necessary
@app.route("/about")
def about_page():
    return "Maybe a page about how it works?"

#submits image to be analyzed
@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        
        dir = 'static/loaded_images'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

        img_path = "static/loaded_images/" +img.filename
        img.save(img_path)
        upload_image("insectImages",img_path,img.filename)
        p,c,m = get_prediction(img_path)
        c = c.title()

    return render_template("index.html", prediction=p, cmnName = c, maybe = m, img_path=img_path)


if __name__ == '__main__':
    app.run()
