# Lyft Challenge Readme

This is a very brief Readme, that describes my approach to the Lyft Challenge. 

## Network
I utilised a Fully Convolutional Network, based on VGG16. I downloaded pretrained weights, and augmented the original VGG16 so that it could be utilsed for Semantic Segmentation, by utilising 1x1 convolutions, upsampling and skip connections. Additionally, I also built a UNet implementation, and trained this on my local PC (VGG16 exhausted my poor 4Gb GPU). Based on my training, UNet resulted in inferior accuracy relative to VGG16, so I chalked it up to a good learning experience, and focused on VGG16.

## Data
I utilised three data sources:
1. https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz
2. https://www.dropbox.com/s/1etgf32uye2iy8q/world_2_w_cars.tar.gz
3. https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180528.zip

I created a shell script to manage the datasets. VGG16 was pretrained on ImageNet, so I made sure to feed RGB images, as opposed to BGR. The images were resized to (160, 288, 3).

# Training
I originally setup my training pipeline to save the model every 10 epochs, as well as run inference inbetween and save the images, to visualise the results. After comparing the performance at 10 epochs to 20 epochs, it appeared that the prediction was becoming worse, so I simply ended the training and my final submitted model was based on 10 epochs. Ideally, I would liked to have trained to 100 epochs, and had time to look through the results, and tweak where possible. However, I was training on a mix of AWS and the Udacity workspace, because my GPU memory is insufficient for local training. Some key things to highlight:
1. I utilised an Adam optimiser with a learning rate of 0.0001
2. I set my dropout to 0.8
3. I utilised a kernel initialiser with a random normal with a std dev of 0.001
4. I utilised a kernel regulariser with l2 regularization with a scale of 0.001
5. My optmiser was minimising the loss of the cross_entropy and regularization losses, based on tf.reduce_mean

# Inference
After training, I converted my checkpoint file to a frozen_graph based on a script I wrote. Then for inference, I invoked the frozen_graph.pb and fed the image into my prediction tensor. I managed to achieve 6.7 FPS in my inference. My next steps are to apply quantization using TensorRt, and then upload it to my TX2 developer kit!
