""" USAGE
python test_network.py --model stop_not_stop.model --image examples/stop001.jpg

This script classfies an input image as stop or not stop using a traned CNN


"""

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


def preprocess_image(image_path):
    """Load and preprocess the image for classification."""
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Image not found at {image_path}")

    orig = image.copy()
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image, orig


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# Load and preprocess the image
image, orig = preprocess_image(args["image"])

# Load the trained convolutional neural network
print("[INFO] Loading network...")
model = load_model(args["model"])


# Build the label
label = "Stop" if stop > notStop else "Not Stop"
proba = stop if stop > notStop else notStop
label = "{}: {:.2f}%".format(label, proba * 100)

# Draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)

# Press 'q' to close the window
if key ==ord('0'):
    cv2.destroyAllWindows()
