import numpy as np
import tensorflow as tf
import cv2
from urllib.request import urlopen
import requests

# Load the saved model for inference
model_path = 'C:/Users/maxvr/PycharmProjects/pythonProject2/archive (1)/Output/emotion_model22.h5'  # Update this to your model's path
model = tf.keras.models.load_model(model_path)

esp32_ip = "..."


def crop_frame(frame):
    # Get dimensions of the frame
    height, width, _ = frame.shape

    # Determine the size of the square region to crop
    crop_size = min(height, width)

    # Calculate coordinates for cropping around the center
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    # Crop the frame
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    return cropped_frame

def update_smiley(happiness):
    url = f'http://{esp32_ip}/?happiness={happiness}'  # Update URL to match the route in Arduino code
    response = requests.get(url)
    if response.status_code == 200:
        print("Smiley updated successfully!")
    else:
        print("Failed to update smiley")

def preprocess_frame(frame, img_width=128, img_height=96):
    # Crop the frame to make it square
    # cropped_frame = crop_frame(frame)

    # Resize the frame to the required dimensions
    frame = cv2.resize(frame, (img_width, img_height))
    # Convert to grayscale
    # Normalize the pixel values
    frame_normalized = frame / 255.0
    # Ensure the data type is float32
    frame_float32 = np.float32(frame_normalized)
    frame_display = np.float32(frame)
    # Expand dimensions to fit model input shape
    frame_expanded = np.expand_dims(frame_float32, axis=-1)  # Add channel dimension
    frame_expanded = np.expand_dims(frame_expanded, axis=0)  # Add batch dimension
    frame_display_expanded = np.expand_dims(frame_display, axis=-1)
    frame_display_expanded = np.expand_dims(frame_display_expanded, axis=0)
    return frame_expanded, frame_display_expanded

# Start capturing video
url = r'...'
while True:
    img_resp = urlopen(url)
    img_data = img_resp.read()

    # Check if image data is empty
    if not img_data:
        print("Empty image data")
        continue

    imgnp = np.asarray(bytearray(img_data), dtype="uint8")
    frame = cv2.imdecode(imgnp, -1)

    # Preprocess the frame
    processed_frame, processed_frame_display = preprocess_frame(frame)

    # Make a prediction
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions)
    confidence_score = np.max(predictions)

    # Display the predicted class and confidence score
    label = f"Class: {predicted_class}, Confidence: {confidence_score:.2f}"
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    update_smiley(predicted_class)

    # Display the combined frame
    cv2.imshow('Combined Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



