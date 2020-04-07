# Coarse-to-Fine Hyper-Prior Modeling for Learned Image Compressionn
## Overview
This is the implementation of ther paper,
> Yueyu Hu, Wenhan Yang, Jiaying Liu, 
> Coarse-to-Fine Hyper-Prior Modeling for Learned Image Compression,
> <i>AAAI Conference on Artificial Intelligence</i> (<i>AAAI</i>), 2020

The currently available code is for evaluation, while it can also be modified for training as the implementation of the network is available.

## Running
The code requires the TensorFlow library (v1.13, v1.14 and v1.15 tested). It should be running in the CPU-only mode, for example, by specifying ```CUDA_VISIBLE_DEVICES= ```. An example to run the encoder and decoder is provided below.

You may first download the trained weights from <a href="https://drive.google.com/open?id=1QL9lpEeTgzJMCEZ2m-9gOxGr6TChB2PU">Google Drive</a> and place the ```.pk``` files under the ```models``` folder (that is, to make ```'./models/model0_qp1.pk``` exist).

### Help
```python AppEncDec.py -h```
### Encoder
```python AppEncDec.py compress example.png example.bin --qp 1 --model_type 0```
### Decoder
```python AppEncDec.py decompress example.bin example_dec.png```

Detailed command line options are documented in the ```help``` mode of the APP.
