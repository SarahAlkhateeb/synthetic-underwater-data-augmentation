# A methodology to detect deepwater corals using Generative Adversarial Networks

## Data
All data and models are published at the Swedish National Data Service under the DOI: https://doi.org/10.5878/hp35-4809.

## Data Augmentation


### Frame Tracking Data Augmentation


### Synthetic Data Augmentation
For StyleGAN2, we used the PyTorch implementation by Zhao et al. [1]. StyleGAN2 can be trained to generate images of size $4\times4$ up to $1024\times1024$, where the sizes double each time. However, since our training images are of size $720\times576$, we trained the network to generate images of size $512\times512$, the highest possible resolution for us. 

The model was trained with the implemented default hyper-parameters: We used an Adam optimizer with momentum parameters ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20B_%7B1%7D%20%3D%200), ![equation](https://www.sciweavers.org/tex2img.php?eq=B_%7B2%7D%3D0.99&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) and learning rate $0.002$ for all weights, except for the mapping network, which used $100$ times lower learning rate. Furthermore, the implementation includes an equalized learning rate approach[^1](#fn1)[2].

For the objective function, StyleGAN2 uses the improved loss from the original GAN paper [3] together with ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20R_1) regularization[^2](#fn1) [4] and regularization parameter ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cgamma%20%3D%2010). As activation function, leaky ReLU was used in both the discriminator and generator with a slope set to ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha%3D0.2).

We trained the network with a batch size of $8$ for a total of $500k$ image iterations (about $1222$ epochs). We chose this training length following the *Low-Shot Generation Experiment* [1] on the AnimalFace dataset [5] with similar size.
During training, we generated images every $40k$ iteration. For the final model, we picked the weights for which the quality of generated images stopped improving. When generating images, the exponential moving average of the generator weights [2] with decay $0.999$ was used to reduce substantial weight variations between training iterations.

## Footnotes
<a id="fn1"></a>[^1]: In the equalized learning rate approach, all weights are initialized from ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathcal%7BN%7D%20%5Csim%20%280%2C1%29) and scaled per-layer using a normalization constant during training. This approach is useful since the weights then have a similar scale during training, and hence, the learning speed is the same for all weights.

<a id="fn1"></a>[^2]: ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20R_1) regularization stabilizes the training process by penalizing the discriminator for deviating from the optimum: ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20R_1%3D%5Cfrac%7B%5Cgamma%7D%7B2%7D%20%5Cmathbb%7BE%7D_%7Bx%5Csim%20%5Cmathbb%7BP%7D_r%7D%5B%5ClVert%5Cnabla%20D%28x%29%20%5CrVert%5E2%5D)

## References

[1] [Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/pdf/2006.10738.pdf)
[2] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)
[3] [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) 
[4] [Which Training Methods for GANs do actually Converge?](https://arxiv.org/pdf/1801.04406.pdf)
[5] [Learning Hybrid Image Templates (HIT) by Information Projection](http://www.stat.ucla.edu/~sczhu/papers/PAMI_HiT.pdf)


## Object Detection
