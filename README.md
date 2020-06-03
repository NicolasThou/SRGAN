# Personal Student Project in Data Science

### Architecture of the folder directory

You have to create the main architecture of the working directory in order to
run the code

``mkdir data results weights``

``cd results``

``mkdir training test prediction``

1) The ``data`` directory is for the celebA dataset
2) The ``results`` directory is for the image generated
    2) ``training`` along the training in order
to visualize how the generator improve
    2) ``prediction`` prediction on a batch of random images
    2) ``test`` test on different images from the test set
3) The ``weights`` directory is for saving and loading the weights for the test and the prediction mode

### Download the dataset celebA manually

You can download the dataset manually via this [link](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM)
and move the file into the directory `data`

Then you have to unzip the file, and remove the `.zip`

### Download the dataset celebA via the terminal

1 - `cd ./data`

2 - `pip install gdown `

3 - `gdown https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM`

Make sure that the downloaded file is in the `data` directory.
Then you have to unzip the file, and remove the `.zip`

### Train the model

### Test the model

### Predict with the model

### Display the loss during training
Go in the terminal in the working directory 

`tensorboard --logdir=runs`

Copy the link given by the terminal : http://localhost:6006/

Paste the link on your browser

