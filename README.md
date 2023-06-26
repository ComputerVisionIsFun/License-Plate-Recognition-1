# License Plate Recognition 

License plate recognition is one of popular applications in the filed of computer vision and artificial intelligence. In General, it is composed of two parts, namely, a plate detector and a classifier of digits and characters. 



In this respository, 



Let $I$ be a batch of images which of size $N\times 3\times H \times W$, where $N$, $H$, and $W$ are batch size, height, width of images, respectively. And let $f$ be the detector composing of several convolutional and LSTM layers. Then f acts on I, we have four outputs, namely, $A, O, C_x, C_y$.