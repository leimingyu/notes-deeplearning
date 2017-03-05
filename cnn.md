### strides, depth and padding

### weights sharing
 We would like our CNNs to also possess this ability known as translation invariance. 
 Recognize objects regardless of their location in the image.
 
 If we want a cat that’s in the top left patch to be classified in the same way as a cat in the bottom right patch, 
 we need the weights and biases corresponding to those patches to be the same, so that they are classified the same way.
 
```
H = height, W = width, D = depth

We have an input of shape 32x32x3 (HxWxD)
20 filters of shape 8x8x3 (HxWxD)
A stride of 2 for both the height and width (S)
Valid padding of size 1 (P)
```

```python
new_height = (input_height - filter_height + 2 * P)/S + 1
new_width = (input_width - filter_width + 2 * P)/S + 1
```
The new height is 14, new width is 14, and new depth is 20.
