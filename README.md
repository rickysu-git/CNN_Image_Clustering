# CNN_Image_Clustering
Extract pretrained CNN features for image clustering. Follows this paper [here](https://arxiv.org/abs/1707.01700#:~:text=These%20results%20strengthen%20the%20belief,approaches%2C%20even%20for%20unsupervised%20tasks.).
Basic steps:
1. Select a pretrained CNN (trained on ImageNet); VGG16 is used for this repo, but choose whichever one your heart desires
2. Remove the last softmax layer (in reality, you can choose whichever layer you want the features to come out of)
3. Gather features for all data points (images)
4. Send the features into a clustering algorithm; K-means is used for this repo, but choose whatever you'd like

## Getting Started
1. Unzip `train_small.zip`

## Dataset
The listed dataset is taken from [here](https://www.kaggle.com/c/dogs-vs-cats). There's 25000 images in the original dataset. Of course, we don't want to blow up Github, so only 100 pictures each of dogs and cats were uploaded, just so you can get started (that should be good enough to get an accuracy of >0.90. Obviously, you can substitute your own dataset.
