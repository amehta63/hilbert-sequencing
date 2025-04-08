# hilbert-sequencing

This code is working up to a hilbert space-filling model that will attempt to predict biosensor fluorescence from sequence data. 

Several tests will be run:
- A 1d convolutional model on the sequence itself
- A 1d convolution model on the sequence and ESM annotations
- A 2d convolutional model on the sequence when fractally transformed into a 2d hilbert curve
- A 3d convolutional model on the 2d sequence above, with a 3rd dimension for ESM annotations (not implemented yet)

On the horizon:
- [Scaling up kernel size](https://arxiv.org/abs/2203.06717) and introducing transposed/global pooling to the 2D network
