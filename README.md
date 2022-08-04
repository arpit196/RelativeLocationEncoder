# RelativeLocationEncoder
The Relative Location Encoder is a CNN model to learn invariant object representations that do not change with Affine Transformations.
The Convolutional Neural Network is a popular model for performing image classification tasks. However, the convolution operation is not invariant to rotations, scaling and affine transformations. In this work, we build a model that is invariant to such affine transformations and encodes the relative orientation between edges
and the distance between parts to account for this invariance, as these features remain invariant to affine transformations. On the CIFAR10 and STL10 dataset, our model suffers less degradation on the test set accuracy than VGG models of similar depth.
