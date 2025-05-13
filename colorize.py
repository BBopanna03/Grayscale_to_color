import numpy as np
import cv2
import os
from cv2 import dnn

# -------- Model file paths --------#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
proto_file = r"D:\Pythonproject\ColorizationProject\Model\colorization_deploy_v2.prototxt"
model_file = r"D:\Pythonproject\ColorizationProject\Model\colorization_release_v2_norebal.caffemodel"
hull_pts = os.path.join(BASE_DIR, "Model", "pts_in_hull.npy")
img_path = r"D:\Pythonproject\ColorizationProject\images\img1.jpg"
# -------- Reading the model params --------#
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)

if net.empty():
    print("Error: Failed to load the Caffe model. Check the file paths and OpenCV version.")


# ----- Reading and preprocessing image --------#
img = cv2.imread(r"D:\Pythonproject\ColorizationProject\images\img1.jpg")
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

scaled = img.astype("float32") / 255.0
lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Resize the image for the network
resized = cv2.resize(lab_img, (224, 224))
L = cv2.split(resized)[0]  # Extract L channel
L -= 50  # Mean subtraction

# Predicting the ab channels from the input L channel
net.setInput(cv2.dnn.blobFromImage(L))
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

# Take the L channel from the image
L = cv2.split(lab_img)[0]
# Join the L channel with predicted ab channel
colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

# Convert the image from Lab to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# Change the image to 0-255 range and convert it from float32 to int
colorized = (255 * colorized).astype("uint8")

# Resize images for visualization
img = cv2.resize(img, (640, 640))
colorized = cv2.resize(colorized, (640, 640))

# Concatenate original and colorized images
result = cv2.hconcat([img, colorized])

# Show image
cv2.imshow("Grayscale -> Colour", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
