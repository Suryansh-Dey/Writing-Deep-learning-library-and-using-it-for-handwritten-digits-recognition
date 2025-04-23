[See it in action](https://drive.google.com/file/d/19v-RnM_f41Dc6QDBhNkDEYHq6BRFmolo/view?usp=sharing)
# Writing-Deep-learning-library-and-using-it-for-handwritten-digits-recognition
Download MNIST DATASET from here http://yann.lecun.com/exdb/mnist/

WARNINGs: 

1 -> NeuralNetwork library uses the convention of storing array of format:
{array_size, element_1,...., element_(array_size-1)};
For eg: if you wanna pass an array like { 1,2,3 } to some function of this library then you have to modify it in above mentioned format which will be like: {4,1,2,3}
Anything else will raise and error or may lead even to segmentation fault!!

2-> If you are using any multi-threaded function of NeuralNetwork library and you have modified nn (returned by nn_create() or nn_load()) after it was passed to nn_thread::constructor(), then you have to call nn_thread::update_nns() before calling that multi-threaded to make nn and nns, stored in nn_thread class, in sink! Once they get it in sink, you will never need to call nn_thread::update_nns() again provided you don't modify nn again.
