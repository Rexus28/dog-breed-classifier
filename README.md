[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Rexus28/dog-breed-classifier/HEAD?urlpath=%2Fvoila%2Frender%2Fdog_breed_classifier_app.ipynb)

## Dog Breed Classifier
# Overview
This jupyter-notebook app (`dog_breed_classifier_app.ipynb`) uses a trained
convolutional neural-network to classify images of dogs by their breed. The
app is based on
[chapter 5](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb)
of the [fastai book](https://github.com/fastai), but its uses the
[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
to focus on dog breeds only. This app is intended as practice and shows a
generalized approach to creating image classifiers using the fastai API.

# Approach
The model is created separately from the app notebook, in the
`dog_breed_classifier.ipynb` notebook.
The image dataset is downloaded directly to the colab notebook environment and
briefly checked to understand the structure by opening a few images. 
The images are organized into directories for each breed, and a regex parser is
included with the fastai DataBlock initialization to parse the breed name
for each label. fastai provides a show batch method, so labels are checked on
a batch of 9 images to make sure the labels are parsed correctly.

All images are presized to 460x460 pixels, so batch transforms can be
applied to randomly crop down to the final size of 224x224. Presizing limits
adverse affects from data augmentations, like rotate and warp, when applied
after cropping. In this case, pixels near the edges could be missing and would
need replaced as blank pixels or interpreted from the image, but either way
is a loss of detail.

The CNN architecture used is a pretrained ResNet34 model. Transfer
learning takes advantage of all the training performed for a state of
the art model via its high quality feature detection. Transfer learning
accelerates and simplifies training without sacrificing model accuracy.

The model is fine tuned and results are checked using the
ClassificationInterpretation class in fastai before cleaning the data. This is
done to get initial results and make sure a working model can be trained
from the data. The source of the dataset is also a factor, because it has been
curated by a reputable source, it's trusted that the data quality is reasonably
high. Another reason is that fastai provides tools for cleaning data based on
model performance. 

The model is checked for issues after the initial training, specifically,
classes it has difficulty with and images with the greatest loss.
The initial model has difficulty with similar breeds: toy and miniature
poodles, husky and Eskimo dog, and American
Staffordshire terrier and Staffordshire bull-terrier. Checking images with
the greatest losses shows most are fine, but there are a few, including one
dog with sunglasses and in some sort of carrier/cage, that are not very
representative of a dog, so it will be cleaned from the dataset. Any images
with more that one breed are removed, and mislabeled images are updated.

# Making Improvements
Fastai provides several tools to help ease the training process. Starting with
the learning rate finder, the model is reinitialized and trained by varying
learning rate on different mini-batches. Plotting losses to learning rate
reveals the ideal rate or range of rates based on the current parameters. A
rate of 6e-3 is chosen, and the head (last layer) is trained for 3 epochs.
After fine tuning the head all layers are unfrozen for training, and the 
learning rate finder is applied again. This time the rate plot shows lower
rates are better, so a rate of 1e-5 is chosen and the model is trained for
5 more epochs. The end error rate is slightly above the initial training, 
which fine-tuned the head for 1 epoch then trained all layers for 3 epochs.

Some experimentation was performed to find the right number of training epochs
while the model is frozen and unfrozen. Training when layers are frozen
(except for the last layer) has greater steps in performance, so these are
favored over training with all layers. The final model is trained
for 10 epochs to fine-tune the head with a learning rate of 6e-3, then
unfrozen and trained for 5 epochs with a range of rates between 1e-6 and 1e-4.
fastai applies discriminate learning rates when given a slice, meaning it
varies the learning rate at each layer. Early layers get a
lower rate than later layers to keep the initial feature detection intact.
The number of epochs is chosen based on training time.

# Results
Validation accuracy reaches 85%, and checking the most confused classes it is
seen that the similar breeds are still causing issues for the classifier. This
dataset also includes images of puppies, which have less distinct breed
features than older dogs, making them a little harder for the model to
classify correctly. In a 2017 paper from Liu et al, the authors achieved an
accuracy of 89% on the Stanford Dogs dataset. Achieving 85% accuracy for this
model is satisfactory, as it is intended as a learning project and only took
a weekend to create.

It is though the limitations of the model are a combination of architecture
capacity and training time. A deeper model, trained well, will provide better
feature detection to classify similar breeds and puppies.
A test with the ResNet50 architecture was performed,
but results were comparable to the ResNet34 version of the model discussed.
The deeper model requires a different training approach that would allow it to
achieve higher accuracy than the shallower model, but that was outside the
scope of this effort.

# Conclusion
The procedure used to load a dataset, create and train a model, then assess and
make improvements is simplified via the fastai API. This makes it easily
adaptable to new problems, datasets, and models. A end-to-end process to
train a dog breed classifier with 85% accuracy was achieved in only a few
hours.

# Using the model
The final model is saved and implemented as a bare-bones web app using
ipywidgets and voila in a jupyter-notebook. The notebook runs in a docker
image via [binder](https://mybinder.org). Click the badge below to open a
link to the app to try it yourself.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Rexus28/dog-breed-classifier/HEAD?urlpath=%2Fvoila%2Frender%2Fdog_breed_classifier_app.ipynb)

# References
@inproceedings{KhoslaYaoJayadevaprakashFeiFei_FGVC2011,
author = "Aditya Khosla and Nityananda Jayadevaprakash and Bangpeng Yao and Li Fei-Fei",
title = "Novel Dataset for Fine-Grained Image Categorization",
booktitle = "First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition",
year = "2011",
month = "June",
address = "Colorado Springs, CO",
}

@inproceedings{,
author = "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei",
title = "ImageNet: A Large-Scale Hierarchical Image Database",
booktitle = "IEEE Conference on Computer Vision and Pattern Recognition",
year = "2009",
}

@book{howard2020deep,
title={Deep Learning for Coders with Fastai and Pytorch: AI Applications Without a PhD},
author={Howard, J. and Gugger, S.},
isbn={9781492045526},
url={https://books.google.no/books?id=xd6LxgEACAAJ},
year={2020},
publisher={O'Reilly Media, Incorporated}
}

