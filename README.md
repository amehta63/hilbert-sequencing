# hilbert-sequencing

This code is working up to a [hilbert space-filling](https://en.wikipedia.org/wiki/Hilbert_curve) model that will attempt to predict biosensor fluorescence from sequence data. 

Several tests will be run:
- A 1d convolutional model on the sequence itself
- A 1d convolution model on the sequence and ESM annotations
- A 2d convolutional model on the sequence when fractally transformed into a 2d hilbert curve
- A 3d convolutional model on the 2d sequence above, with a 3rd dimension for ESM annotations (implemented, but too computationally expensive)

Some choices that were made:
- The protein sequences come in two sizes: 450 and 422. The difference is mostly one 26 AA section at the begining of the sequence. I chose to fill this section with '-' in the shorter sequence. This should be interpreted as 'unk' in ESM, and is still given a vector and changes the embeddings for the other AAs. Unclear if this is the best decision or not.
- AA to int embeddings used ord(single letter code). This should have no effect on learning.

On the horizon:
- [Scaling up kernel size](https://arxiv.org/abs/2203.06717) and introducing transposed/global pooling to the 2D network
