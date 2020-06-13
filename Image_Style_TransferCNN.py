# Import major libraries that can be useful while implementing the below model
import PIL.Image as Image
from io import BytesIO
import requests
import numpy as np

#import keras for CNN model
from keras import backend
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

#import L-BFGS Algorithm library
from scipy.optimize import fmin_l_bfgs_b

#size of input images and output images
IMAGE_WIDTH = 500
IMAGE_HEIGHT = 500

# Defined paths which can be used for saving the images as external
japanese_garder_image_path = "input.png"
picasso_image_path = "style.png"
output_image_path = "output.png"
combined_image_path = "combined.png"


#Defining target image
japanese_garden_image_path = "https://raw.githubusercontent.com/myelinfoundry-2019/challenge/master/japanese_garden.jpg"

#Input visualization
input_image = Image.open(BytesIO(requests.get(japanese_garden_image_path).content))
input_image = input_image.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
input_image.save(japanese_garden_image_path)
input_image.show()

#Defining style Image
picasso_image_path = "https://raw.githubusercontent.com/myelinfoundry-2019/challenge/master/picasso_selfportrait.jpg"

#Style visualization
style_image = Image.open(BytesIO(requests.get(picasso_image_path).content))
style_image = style_image.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
style_image.save(picasso_image_path)
style_image.show()


# Data normalization and reshaping from RGB to BGR
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
input_image_array = np.asarray(input_image, dtype="float32")
input_image_array = np.expand_dims(input_image_array, axis=0)
input_image_array[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
input_image_array[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
input_image_array[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
input_image_array = input_image_array[:, :, :, ::-1]

style_image_array = np.asarray(style_image, dtype="float32")
style_image_array = np.expand_dims(style_image_array, axis=0)
style_image_array[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
style_image_array[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
style_image_array[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
style_image_array = style_image_array[:, :, :, ::-1]


# CNN Model
input_image = backend.variable(input_image_array)
style_image = backend.variable(style_image_array)
combination_image = backend.placeholder()

input_tensor = backend.concatenate([input_image,style_image,combination_image], axis=0)
model = VGG16(input_tensor=input_tensor, include_top=False)


""" DEFINING LOSS FUNCTION """

#CONTENT LOSS 
def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

layers = dict([(layer.name, layer.output) for layer in model.layers])

CONTENT_WEIGHT = 0.02

content_layer = "block2_conv2"
layer_features = layers[content_layer]
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss = backend.variable(0.)
loss += CONTENT_WEIGHT * content_loss(content_image_features,
                                      combination_features)


#STYLE LOSS
def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

CHANNELS = 3
def compute_style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_WIDTH*IMAGE_HEIGHT
    return backend.sum(backend.square(style - combination)) / (4. * (CHANNELS ** 2) * (size ** 2))

style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
STYLE_WEIGHT = 4.5
for layer_name in style_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    style_loss = compute_style_loss(style_features, combination_features)
    loss += (STYLE_WEIGHT / len(style_layers)) * style_loss



#Total Variation LOSS
TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25
def total_variation_loss(x):
    a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
    return backend.sum(backend.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))

loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)


#Optimization on each iteration to reduce or minimize the loss
outputs = [loss]
outputs += backend.gradients(loss, combination_image)

def evaluate_loss_and_gradients(x):
    x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outs = backend.function([combination_image], outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients

class Evaluator:

    def loss(self, x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients

evaluator = Evaluator()


#RESULTS
x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.

""" Defining iterations, more the number of iterations, more will be the image style transfer.
    For the GPU setting and due to time constraint, the iteration we are providing here is 10,
    but you are allowed to change the number of iteration according to your GPU power. """
    
ITERATIONS = 10
for i in range(ITERATIONS):
    x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
    print("Iteration %d completed with loss %d" % (i, loss))

x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
x = x[:, :, ::-1]
x[:, :, 0] += IMAGENET_MEAN_RGB_VALUES[2]
x[:, :, 1] += IMAGENET_MEAN_RGB_VALUES[1]
x[:, :, 2] += IMAGENET_MEAN_RGB_VALUES[0]
x = np.clip(x, 0, 255).astype("uint8")
output_image = Image.fromarray(x)
output_image.show()


# Visualizing combined results
""" It will show the combined images of input image, style image and the output image
            (the images that are saved as external) """

combined_image = Image.new("RGB", (IMAGE_WIDTH*3, IMAGE_HEIGHT))
x_offset = 0
for image in map(Image.open, [input_image_path, style_image_path, output_image_path]):
    combined_image.paste(image, (x_offset, 0))
    x_offset += IMAGE_WIDTH
combined.save(combined_image_path)
combined_image.show()