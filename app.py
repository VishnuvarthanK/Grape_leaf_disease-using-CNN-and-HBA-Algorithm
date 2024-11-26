import os
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn

# Define the CNN model class (same as during training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)  # Adjust based on image size after pooling
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)  # Flatten the output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # Sigmoid activation for binary classification
        return x

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
MODEL_PATH = "models/cnn_model.pth"  # Ensure the correct path to the .pth file
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load model weights
model.eval()  # Set model to evaluation mode

# Define possible diseases
disease_classes = ["healthy grape leaf", "infected grape leaf"]

# Define image transformations for preprocessing (resize, normalization)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use same mean & std as pre-trained models
])

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    """Handle image upload and prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]

    # Save the file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Preprocess the image
    image = Image.open(file_path).convert('RGB')  # Open the image and convert to RGB
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

    # Make a prediction
    with torch.no_grad():  # No need to track gradients for inference
        prediction = model(image).item()  # Get the prediction score
    predicted_class = disease_classes[int(prediction > 0.5)]  # Threshold at 0.5 for binary classification
    probability = prediction if prediction > 0.5 else 1 - prediction

    return jsonify({
        "predicted_disease": predicted_class,
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)
