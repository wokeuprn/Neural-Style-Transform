#  Neural Style Transfer with PyTorch
## Description
This project implements Neural Style Transfer using PyTorch and a pretrained VGG19 convolutional neural network. The goal of neural style transfer is to blend the content of one image with the style of another, producing a new, visually stunning result.

The application loads two images — a content image and a style image — and optimizes a third image to preserve the content of the former and the artistic style of the latter. This is achieved by minimizing content and style losses computed from the CNN's intermediate layers.

## Example Use Case
* 📷 Content Image: man.jpg
* 🎨 Style Image: city.jpg
* 🧠 Output: A stylized version of the content image with the style patterns of the city

## 🔧 Features
* Uses pretrained VGG19 from torchvision.models
* Supports GPU acceleration if CUDA is available
* Fully customizable: content and style weights, number of iterations
* Visual output using matplotlib
* Losses:
    * Content Loss via MSE
    * Style Loss using Gram Matrix comparison

## 🧠 Technologies Used
* Python 3.9+
* PyTorch
* Torchvision
* PIL for image processing
* Matplotlib for visualization

## 🚀 How to Run
1) Install dependencies
   ````
   pip install torch torchvision matplotlib pillow
2) Add your images
    * Replace man.jpg with your content image
    * Replace city.jpg with your style image

3) Run the script
   ````
    python neural_style_transfer.py
4) View the output
   * The output image will be displayed using matplotlib
   * You can also save it by modifying the final section of the code

## ⚙️ Customization
**You can tweak the following parameters in run_style_transfer():**
  * num_steps: Number of optimization steps (default: 300)
  * style_weight: How strongly to apply the style (default: 1e6)
  * content_weight: How much content to retain (default: 1)


