### strides, depth and padding

same vs valid padding


### parameters sharing
 We would like our CNNs to also possess this ability known as translation invariance. 
 Recognize objects regardless of their location in the image.
 
 If we want a cat that’s in the top left patch to be classified in the same way as a cat in the bottom right patch, 
 we need the weights and biases corresponding to those patches to be the same, so that they are classified the same way.
 
### Output Shape
```
H = height, W = width, D = depth

We have an input of shape 32x32x3 (HxWxD)
20 filters of shape 8x8x3 (HxWxD)
A stride of 2 for both the height and width (S)
Valid padding of size 1 (P)
```

```
new_height = (input_height - filter_height + 2 * P)/S + 1
new_width = (input_width - filter_width + 2 * P)/S + 1
```
The new height is 14, new width is 14, and new depth is 20.
```python
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'VALID'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```
The output shape of conv will be [1, 13, 13, 20]. 
It's 4D to account for batch size, but more importantly, it's not [1, 14, 14, 20]. 
This is because the padding algorithm TensorFlow uses is not exactly the same as the one above.
https://www.tensorflow.org/api_guides/python/nn#Convolution


### parameters number
**without parameters sharing**
```
Setup
H = height, W = width, D = depth

We have an input of shape 32x32x3 (HxWxD)
20 filters of shape 8x8x3 (HxWxD)
A stride of 2 for both the height and width (S)
Zero padding of size 1 (P)
Output Layer
14x14x20 (HxWxD)
```
Without parameter sharing, each neuron in the output layer must connect to each neuron in the filter. 
In addition, each neuron in the output layer must also connect to a single bias neuron.
```
(8 * 8 * 3 + 1) * (14 * 14 * 20) = 756560
```

**with parameters sharing**

With parameter sharing, each neuron in an output channel shares its weights with every other neuron in that channel. 
So the number of parameters is equal to the number of neurons in the filter, plus a bias neuron, 
all multiplied by the number of channels in the output layer.
```
(8 * 8 * 3 + 1) * 20 = 3840 + 20 = 3860
```

### visualizing cnn
* https://www.youtube.com/watch?v=ghEmQSxT6tw
* http://www.matthewzeiler.com/pubs/arxive2013/eccv2014.pdf



