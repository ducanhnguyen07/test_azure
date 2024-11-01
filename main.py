from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import io

# Load the VGG16 model (replace with ResNet50 if appropriate)
model = load_model("model.resnet50.h5")
output_class = ["battery", "glass", "metal", "organic", "paper", "plastic"]

# Initialize FastAPI app
app = FastAPI()

def preprocess_image(img):
  img = img.resize((224, 224))  # Resize to 224x224
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_input(img)  # Use VGG16 preprocess_input
  return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  # Check if the uploaded file is an image
  if file.content_type not in ["image/jpeg", "image/png"]:
    raise HTTPException(status_code=400, detail="File format not supported")
 
  try:
    # Read and process the image
    img = Image.open(io.BytesIO(await file.read()))
    preprocessed_img = preprocess_image(img)

    # Make prediction
    prediction = model.predict(preprocessed_img)
    predicted_class = output_class[np.argmax(prediction)]
    predicted_accuracy = round(np.max(prediction) * 100, 2)

    return {
      "predicted_class": predicted_class,
      "accuracy": predicted_accuracy
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post('/test')
def test():
  return {"message": "Hello World"}