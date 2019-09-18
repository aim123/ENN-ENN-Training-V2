# ENN Training

## Overview

ENN stands for the Evolution of Neural Networks.

The client provided here works by asking a service (source not included here)
for a population of neural network architectures in [Keras 2.2.4](https://keras.io/layers/about-keras-layers/)
JSON format.

Upon receving these network descriptions, the client then sends
out requests for each of the candidate networks to be trained for a small
number of epochs by a (possibly remotely distributed) worker process.

After measurements of each candidate network's performance come back,
fitness measurements are sent back to the same ENN service, where the 
candidate networks are used as a basis to create another new, evolved
generation of candidate networks.

## Goal of Training
Demo below Use Cases:
    1.Image Classification
        i.Chest X-ray Classification ( Multi- Label)
        ii.UTK Face Age Classification ( Multi- Class) 
    2.Text Classification
        i.Toxicity Classification ( Binary)
        ii.Survey Classification ( Multi- Output)
        
#Dataset
for UTKFace data refer: https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE

