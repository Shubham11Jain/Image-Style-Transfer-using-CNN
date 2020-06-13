# Image-Style-Transfer-using-CNN
This assignment is based on a Deep Learning Model - CNN which is used for the image style transfer.

Problem
Objective is to apply the style of an image, which we will term as "style image" to a target image while preserving the content of the target image.

Style is textures and visual patterns in an image. Example is brush strokes of an artist
Content is the macro structure of an image, like buildings, people, objects in the content of the image.

High Level Steps
choose the image to style
choose the style reference image (here we have provided a Picasso style image)
choose a pre-trained deep neural network (CNN type) and obtain feature representations of intermediate layers. This step is done to achieve the representations of both the content image and style image. For the content image, best option is to obtain the feature representations of highest layers as they would contain information on the image macro-structure. For the style reference image, feature representations are obtained from multiple layers at different scales.
define loss function
Loss function should be taking into account the Content-loss, Style-loss, Variation-loss.
Optimize on each iteration to minimize the loss

Recommended Overview of the steps generally followed
Create a random input image
Pass the input through a pre-trained backbone architecture
Calculate loss and compute the gradients w.r.t input image pixels. Hence only the input pixels are adjusted whereas the weights remain constant.
