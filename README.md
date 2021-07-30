# Spatial-Transformer-Network
Network augmentation using visual attention mechanism for spatial transformation on CIFAR10


#### Link to Colab
https://colab.research.google.com/drive/1lTkkla-rlLvLR7epnNdJAm_S7T4c5i62?usp=sharing



https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html<br/>
https://arxiv.org/abs/1506.02025<br/>


#### SPATIAL TRANSFORMER NETWORK

> However, current CNN models still exhibit a poor ability to be invariant to spatial transformations of images.


While CNN architectures are great at translation invariance, they perform poorly when it comes to spatial transformations. 

When we perform image augmentation, we make training a little difficult for the model. 

This is because as the training data becomes more difficult for the model to learn, the better the model
performs on the test set. The different augmentations added to the images are transformations. 

Spatial Transformation Network takes a different route and may be thought of as an opposite strategy. Rather than make
the model work harder, it makes the work of the model simpler by attempting to project the object into a simpler spatial form. 

This works especially well because rather than specifying fixed transformations as is typically done in image augmentation, 
we allow the network to learn the best way to perform the transformations that work best for that particular problem since
it may be inserted in the model architecture. 


[![image.png](https://i.postimg.cc/k429vB52/image.png)](https://postimg.cc/R6xjMCKm)


STN is broken down into 3 components - 

#### Localization Network
This takes the input image, or feature map and outputs a transformation parameter (theta) 
to be applied to the input. 


```python
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
```


#### Grid Generator
Takes the output of the localization nework and generates grid.

```python
grid = F.affine_grid(theta, x.size())
```


#### Differentiable Image Sampler
Performs spatial transformation of the input feature map using the grid 
generated to produce an output feature map. 


```python
x = F.grid_sample(x, grid)
```


#### How to run

> run main.py 




### Reference
Xu Shen, Xinmei Tian, Anfeng He, Shaoyan Sun, Dacheng Tao 2019 Transform-Invariant Convolutional Neural Networks for Image Classification and Search https://arxiv.org/abs/1912.01447
