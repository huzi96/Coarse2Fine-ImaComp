# The implementation of GDN is inherited from
# https://github.com/tensorflow/compression,
# under the Apache License, Version 2.0. The 
# source code also include an implementation
# of the arithmetic coding by Nayuki from
# https://github.com/nayuki/Reference-arithmetic-coding
# under the MIT License.
#
# This file is being made available under the BSD License.  
# Copyright (c) 2020 Yueyu Hu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob

import arithmeticcoding

import numpy as np
import tensorflow as tf
import pickle

import math
import time

from networks import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["compress", "decompress"],
      help="What to do? Choose from `compress` and `decompress`")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")
  parser.add_argument(
      "--qp", default=1, type=int,
      help="Quality parameter, choose from [1~7] (model0) or [1~8] (model1)"
  )
  parser.add_argument(
      "--model_type", default=0, type=int,
      help="Model type, choose from 0:PSNR 1:MS-SSIM"
  )
  parser.add_argument(
      "--save_recon", default=True, type=bool,
      help="Whether to save reconstructed image in the encoding process."
  )

  args = parser.parse_args()

  if args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    if args.qp <= 3:
      compress_low(args)
    else:
      compress_high(args)
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    decompress_low(args)
    decompress_high(args)
