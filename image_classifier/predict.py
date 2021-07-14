import argparse
import json
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid warning

# Parse arguments
parser = argparse.ArgumentParser(
    description='Predict flowers from images',
)

parser.add_argument('image_file', action="store", help="Image file to predict")
parser.add_argument('model_file', action="store", help="Model to use for predictions (the trained model)")
parser.add_argument('--top_k', action="store", type=int, default=1, help="Top number of categories to show")
parser.add_argument('--category_names', action="store", help="Name of file with category names")

args = parser.parse_args()

# Define functions

def process_image(image):
    image_size = 224
    res = tf.convert_to_tensor(image, np.float32)
    res = tf.image.resize(res, (image_size, image_size))
    res /= 255
    return res.numpy()

def predict(image_path, model, top_k, names = None):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.expand_dims(image, 0)
    image = process_image(image)
    res = model.predict(image)[0]
    top = np.flip(res.argsort()[-top_k:])
    cat = []
    for t in top:
        if names:
            cat.append(class_names[str(t + 1)])
        else:
            cat.append(str(t + 1))
    return res[top], cat

# Main

reloaded_model = tf.keras.models.load_model(args.model_file, custom_objects={'KerasLayer':hub.KerasLayer})

class_names = None
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

probs, classes = predict(args.image_file, reloaded_model, args.top_k, class_names)

print(f'\nThe picture most probably shows a {classes[0]} (with {probs[0] * 100:.2f}% certainty)\n')

if args.top_k > 1:
    print(f'The top {args.top_k} most probable flowers:\n')
    for i in range(args.top_k):
        print(f'{i + 1:2}. {classes[i]:30}   {probs[i] * 100:.2f}%')
else:
    print("Use the --top_k option to see more than one alternative")

print("\n")

if not class_names:
    print("Use the --category_names option to show names for the flowers")
    print("This may work well:")
    print("  --category_names label_map.json")
    print("\n")

