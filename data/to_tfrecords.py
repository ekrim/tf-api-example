import struct
import os
import numpy as np


def cifar_reader(f):
  byte_str = f.read(3073) 
  if byte_str==b"":
    return None
  else:
    record_tuple = struct.unpack("3073B", byte_str)
    label = record_tuple[0]
    img = np.array(record_tuple[1:]).astype(np.uint8)
    img = np.concatenate(np.split(img.reshape((32*3,32))[:,:,None], 3, axis=0), axis=2)
    return img

if __name__=="__main__":

  with open(fn, "rb") as f:
    while  
    cifar_reader(f)  
