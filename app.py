#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on: August 16 ,2024
@author: somiljain7
"""
from flask import Flask, request, redirect, render_template, flash, send_file
import subprocess
import wave
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yolov8 import YOLOv8
import shutil 
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

app = Flask(__name__)
app.secret_key = "Somil"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import image_object_detection as obd
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.45, iou_thres=0.5)
import json

def numpy_to_json(np_array, file_name):
    data = []
    for row in np_array:
        entry = {
            "X1": row[0],
            "y1": row[1],
            "X2": row[2],
            "Y2": row[3],
            "scores": row[4]
        }
        data.append(entry)
    
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data successfully saved to {file_name}")


def img_plot(img_list,image_titles):
    # Create subplots with the correct number of rows and columns
    num_images = len(img_list)
    fig = make_subplots(
        rows=1, cols=num_images,
        subplot_titles=image_titles,
        horizontal_spacing=0.005
    )

    for i, img in enumerate(img_list):
        fig.add_trace(
            go.Image(z=img),
            row=1, col=i+1
        )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=30, b=0),
        width=400 * num_images,
        height=300)

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    fig_html = fig.to_html(full_html=False)
    return fig_html

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            file = request.files["file"]
            f_name = file.filename
            tp = ""
            if f_name == "":
                file_name = request.form["EXAMPLE_IMAGES"]
                f_name = file_name
                tp = "demo"
            else:
                file.save(os.path.join("exp", f_name))
                tp = "upload"

            submit = request.form["submit"]
            image_path = os.path.join(os.getcwd(), "exp", f_name)
            pt6 = os.path.splitext(f_name)[0] + ".jpg"

            if submit == "process":
                bounding_boxes, scores, class_ids = obd.Objectdetection_yolo8(yolov8_detector, image_path)
                img = obd.grouping(bounding_boxes, image_path)
                image_og= np.array(Image.open(image_path))
                Object_yolo_output=np.array(Image.open("exp/detected_objects.jpg"))
                #print(np.array(scores).shape,np.array(bounding_boxes).shape)
                # Combine all images in a list
                images = [image_og, Object_yolo_output, img]
                image_titles=['Orignal Image','Image with bboxes','Grouping of labels']
                bbox=bounding_boxes.tolist()
                scores=scores.tolist()
                np_array=np.array([bbox[i] + [scores[i]] for i in range(len(scores))])
                numpy_to_json(np_array, 'exp/output.json')
                # Generate the combined figure
                fig_html = img_plot(images,image_titles)

                required_files = ["output.json", "detected_objects.jpg", "detected_objects_clustered.jpg"]
                if all(os.path.isfile(os.path.join("exp", file)) for file in required_files):
                    return render_template("UPLOAD.html", fig_html=fig_html)
        return render_template("main.html")
    except Exception as e:
        print(e)
        return render_template("error.html")


@app.route('/download-json')
def download_json():
    json_file_path = os.path.join("exp", "output.json")  # Adjust the path to your JSON file
    return send_file(json_file_path, as_attachment=True, download_name='output.json')

@app.route('/success')
def success():
    return 'File successfully processed by  script'


                  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

