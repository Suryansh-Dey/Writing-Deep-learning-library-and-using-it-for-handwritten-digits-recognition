# Writing-Deep-learning-library-and-using-it-for-handwritten-digits-recognition
WARNINGs: 

1 -> NeuralNetworks library uses the convention of storing array of format:
{array_size, element_1,...., element_(array_size-1)};
For eg: if you wanna pass an array like { 1,2,3 } to some function of this library then you have to modify it in above mentioned format which will be like: {4,1,2,3}
Anything else will raise and error or may lead even to segmentation fault!!

2-> If you are using any multi-threaded function of this library and you have modified nn (returned by nn_create() or nn_load()) after it was passed to nn_thread::constructor, then you have to call nn_thread::update_nns() before calling that multi_threaded to make nn and nns, stored in nn_thread class, in sink! Once they get it in sink, you will never need to call nn_thread::update_nns() again provided you don't modify nn again.
