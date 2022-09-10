# newestinsect
This repository contains updated code for an insect classification web application for the Spark Development Challenge.
The image processing is performed using OpenCV.
The image classification is performed using a PyTorch Model.
The backend of the website uses Flask.
The frontend uses HTML, CSS, and Javascript.

1. This project takes an image of an insect and classifies it using a PyTorch Model. I previously created the PyTorch Model for another project I was working on, so I thought the Spark challenge would be a nice opportunity to actually integrate it into a web app.

2. I chose to link a frontend framework (HTML, CSS, JavaScript) with a Flask backend. I also based my frontend on viewport width, so it is responsive to mobile dimensions. I have API calls in order to process the multipart HTTP post requests for image transfer and return predictions. In addition, I save the images uploaded to the website to Google Cloud Storage. Moreover, I deployed my website to a Google Virtual Machine in order to quickly scale to higher volume of requests just in case everyone at Spark is extremely intrigued by my site :).

3. I spent about 3 hours developing the web app, but I also created an accompanying mobile app and REST API, which took me an extra hour since I used the base code for another mobile app that I built. The repo for the mobile app is https://github.com/Sahish08/insect_mobile_app and the REST API is https://github.com/Sahish08/flask-api.

4. My project can be accessed at the following IP address: 35.226.90.78
