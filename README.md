# Personal Student Project in Data Science

### Architecture of the folder directory

You have to create the main architecture of the working directory in order to
run the code

``mkdir data results weights``

``cd results``

``mkdir training test prediction``


1) The ``data`` directory is for the celebA dataset. 
    a) ``img_align_celeba`` folder that contains every images
2) The ``results`` directory is for the image generated
    a) ``training`` along the training in order
to visualize how the generator improve
    b) ``prediction`` prediction on a batch of random images
    c) ``test`` test on different images from the test set
3) The ``weights`` directory is for saving and loading the weights for the test and the prediction mode

### Download the dataset celebA manually

BEFORE, remove the already existing folder ``img_align_celeba`` in the folder ``data``

You can download the dataset manually via this [link](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM)
and move the file into the directory `data`

Then you have to unzip the file, and remove the `.zip`

### Download the dataset celebA via the terminal

BEFORE, remove the already existing folder ``img_align_celeba`` in the folder ``data``

1 - `cd ./data`

2 - `pip install gdown `

3 - `gdown https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM`

Make sure that the downloaded file is in the `data` directory.
Then you have to unzip the file, and remove the `.zip`

### Train the model

To train the model, just run the file `train.py`

### Test the model

To train the model, just run the file `test.py`

### Predict with the model

To make a random prediction with the model, just run the file `predict.py`

### Display the loss during training
Go in the terminal in the working directory 

`tensorboard --logdir=runs`

Copy the link given by the terminal : http://localhost:6006/

Paste the link on your browser

