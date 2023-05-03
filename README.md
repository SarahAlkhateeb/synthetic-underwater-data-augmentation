# A methodology to detect deepwater corals using Generative Adversarial Networks

## Data
All data and models are published at the Swedish National Data Service under the DOI: https://doi.org/10.5878/hp35-4809.

## Data Augmentation


### Frame Tracking Data Augmentation


### Synthetic Data Augmentation
For StyleGAN2, we used the PyTorch implementation by [1]. StyleGAN2 can be trained to generate images of size $4\times4$ up to $1024\times1024$, where the sizes double each time. However, since our training images are of size $720\times576$, we trained the network to generate images of size $512\times512$, the highest possible resolution for us. 

The model was trained with the implemented default hyper-parameters: We used an Adam optimizer with momentum parameters $\beta_1=0$, $\beta_2=0.99$
and learning rate $0.002$ for all weights, except for the mapping network, which used $100$ times lower learning rate. Furthermore, the implementation includes an equalized learning rate approach[^1](#fn1) [2].

For the objective function, StyleGAN2 uses the improved loss from the original GAN paper [3] together with $R_1$ regularization[^2](#fn1) [4] and regularization parameter $\gamma = 10$. As activation function, leaky ReLU was used in both the discriminator and generator with a slope set to $\alpha=0.2$.

We trained the network with a batch size of $8$ for a total of $500k$ image iterations (about $1222$ epochs). We chose this training length following the *Low-Shot Generation Experiment* [1] on the AnimalFace dataset [5] with similar size.
During training, we generated images every $40k$ iteration. For the final model, we picked the weights for which the quality of generated images stopped improving. When generating images, the exponential moving average of the generator weights [2] with decay $0.999$ was used to reduce substantial weight variations between training iterations.

## Footnotes
<a id="fn1"></a>[^1]: In the equalized learning rate approach, all weights are initialized from $\mathcal{N} \sim (0,1)$ and scaled per-layer using a normalization constant during training. This approach is useful since the weights then have a similar scale during training, and hence, the learning speed is the same for all weights.

<a id="fn1"></a>[^2]: $R_1$ regularization stabilizes the training process by penalizing the discriminator for deviating from the optimum: $R_1=\frac{\gamma}{2} \mathbb{E}_{x\sim \mathbb{P}_r}[\lVert\nabla D(x) \rVert^2]$}

## References

[1] [Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/pdf/2006.10738.pdf)

[2] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)

[3] [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) 

[4] [Which Training Methods for GANs do actually Converge?](https://arxiv.org/pdf/1801.04406.pdf)

[5] [Learning Hybrid Image Templates (HIT) by Information Projection](http://www.stat.ucla.edu/~sczhu/papers/PAMI_HiT.pdf)


## Object Detection
