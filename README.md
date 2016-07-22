## Fetch and load dataset to Python/NumPy

`datasetfetch` provides easy access to a collection of machine learning datasets by automatically downloading, unpacking and loading them into Python.
The datasets are typically loaded into Python as NumPy arrays.
If the dataset are too large to fit in memory, functionality is provided to access parts of the dataset.

We leave it to the user to preprocess and feed the data efficiently to the machine learning pipeline.

Currently, the following datasets are available through `datasetfetch`:

 * [Large-scale CelebFaces Attributes (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
 * [CIFAR-10 and CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)
 * [Infinite MNIST (formerly known as MNIST8M)](http://leon.bottou.org/projects/infimnist)
 * [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
 * [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/)
 * [Multi-view Stereo Correspondence Dataset](http://cs.ubc.ca/~mbrown/patchdata/patchdata.html)
 * [The STL-10 dataset](http://cs.stanford.edu/~acoates/stl10)
