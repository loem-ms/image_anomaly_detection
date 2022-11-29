import os

import numpy as np
from model import ResNetVAE
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import KernelDensity

from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../frontend/app/src/media'
CORS(app)

encoded_dim = 64
ffn_hidden_dim = 512

model = ResNetVAE(encoded_dim, ffn_hidden_dim * 2, ffn_hidden_dim)
model.load_state_dict(torch.load("./checkpoints/model_resnet_scratch190.pth", map_location=torch.device('cpu')))
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize(900),
    transforms.ToTensor(),
])

encoded_good_vectors = np.loadtxt('./checkpoints/encoded_traindata_pill.csv', delimiter=',')
kernel_density = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(encoded_good_vectors)
threshold = np.min(
    kernel_density.score_samples(encoded_good_vectors)
)

@app.route("/", methods=['GET'])
def index():
    return "Anomaly Detection"

@app.route("/detection", methods=['GET', 'POST'])
def detect_anomaly():
    save_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded')
    restored_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'restored')
    
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    if not os.path.isdir(restored_folder):
        os.mkdir(restored_folder)
        
    try:
        img_file = request.files['imageFile']
        filename = secure_filename(img_file.filename)
        destination = "/".join([save_folder, filename])
        #restored_dest = "/".join([restored_folder, filename])
        img_file.save(destination)
        
        label, _ = check_anomaly(destination)
        response = {'label': f"Result: {label}"}
    except:
        response = {'label': f"No image file selected/Unvalid file"}
        
    return response

def check_anomaly(image_path):
    image = Image.open(image_path)
    image = image_transforms(image)
    image = image.unsqueeze(0)
    encoded_image = model.encode(image)
    encoded_vector = encoded_image.reshape(1, -1)
    encoded_density = kernel_density.score_samples(encoded_vector.detach().numpy())
    
    label = "Has defect" if encoded_density[0] < threshold else "No defect"
    
    return label, encoded_density

if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port=5000)