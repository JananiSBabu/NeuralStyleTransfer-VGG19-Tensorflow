import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
from nst_helpers import *
import numpy as np
import tensorflow as tf
import argparse

# =========================================================
# Collect arguments from cmd line and parse them
ap = argparse.ArgumentParser(description="Neural style transfer Application",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--content_image_filename", help="Content image filename", default="louvre_small.jpg", type=str)
ap.add_argument("--style_image_filename", help="Style image filename", default="monet.jpg", type=str)
ap.add_argument("--epochs", help="Number of epochs for training", default=15, type=int)
ap.add_argument("--print_every", help="Print cost/generated image every", default=20, type=int)
ap.add_argument("--alpha", help="hyperparameter weighting the importance of the content cost", default=10, type=float)
ap.add_argument("--beta", help="hyperparameter weighting the importance of the style cost", default=40, type=float)
ap.add_argument("--learning_rate", help="Learning rate for optimizer", default=2.0, type=float)
args = vars(ap.parse_args())

# consume user inputs
epochs = args["epochs"] # 200
print_every = args["print_every"] # 20
content_image_filename = "images/" + args["content_image_filename"]
style_image_filename = "images/" + args["style_image_filename"]
alpha = args["alpha"]
beta = args["beta"]
learning_rate = args["learning_rate"]

print("\n\n\n================================================")
print("content_image_filename : ", content_image_filename)

print("style_image_filename : ", style_image_filename)
# =========================================================


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

# Load the Content image
content_image = scipy.misc.imread(content_image_filename)
content_image = reshape_and_normalize_image(content_image)

# Load the Style Image
style_image = scipy.misc.imread(style_image_filename)
style_image = reshape_and_normalize_image(style_image)

# Initialize Generated image with Noise and a small correlation with Content image
generated_image = generate_noise_image(content_image)

# Load the pre-trained VGG-19 model
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

# Assign the content image to be the input of the VGG model.
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS, sess)

# Compute the total cost
J_total = total_cost(J_content, J_style, alpha=alpha, beta=beta)

# Compute the total variation lost - to get smoother results
total_variation_weight = 2.0
J = J_total + total_variation_weight * tf.image.total_variation(a_G)

# define optimizer (Adam)
optimizer = tf.train.AdamOptimizer(learning_rate)

# define train_step
train_step = optimizer.minimize(J)

# Build the model by training over several Epochs

# Initialize global variables
sess.run(tf.global_variables_initializer())

# Run the noisy input image (initial generated image) through the model. Use assign().
sess.run(model["input"].assign(generated_image))

for i in range(epochs):

    # Run the session on the train_step to minimize the total cost
    sess.run(train_step)

    # Compute the generated image by running the session on the current model['input']
    generated_image = sess.run(model["input"])

    # Print cost and save generate image for "print_every" iterations.
    if i % print_every == 0:
        Jt, Jc, Js = sess.run([J, J_content, J_style])
        print("Iteration " + str(i) + " :")
        print("total cost = " + str(Jt))
        print("content cost = " + str(Jc))
        print("style cost = " + str(Js))

        # save current generated image in the "/output" directory
        save_image("output/" + str(i) + ".png", generated_image)

# save last generated image
save_image('output/generated_image.jpg', generated_image)