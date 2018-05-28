# Election-Prediction
A Keras model to predict who will win Donald Trump or Hillary Clinton in a county depending on various features like
'Per capita income' , 'Total hoseholds' , 'Unemployemnt rate' , etc. in that county.

# Datasets
Datasets used are in the folder 'data'.

# Requirements

1.) Keras with Tensorflow backend

2.) Numpy

3.) Pandas

# How to Run

To execute the code, just run the main.py file. It will call the required model for training and testing.

# Results

Statistics at Epoch 100/100: loss: 0.2149, acc: 0.7580, val_loss: 0.2139, val_acc: 0.8361

It is observed that increasing the number of epochs increases accuracy and further decreases loss.

The dropouts also allowed significant improvement (by several %). This is because with using Dropouts, I was able to avoid overfitting.

I also set differing number of units when calling Dense() on each layer. This adjusted the dimension of the output. I noticed that when I steadily increase, then decrease the number of units across my layers the accuracy improved.

Through these changes and adjustments, my validation accuracy went up to around 83% with 100 epochs. It is important to note that the training accuracy stayed at around 76%. This is due to the Droputs. Therefore, it is a good sign that the validation accuracy is higher than the training accuracy: this proves that Dropouts alone improve the accuracy by several %.

# Images

### Epoch 1-4
![alt txt](https://github.com/kshitij1210/Election-Prediction/blob/master/images/epoch1-4.JPG)

### Epoch 5-8
![alt txt](https://github.com/kshitij1210/Election-Prediction/blob/master/images/epoch5-8.JPG)

### Epoch 9-10
![alt txt](https://github.com/kshitij1210/Election-Prediction/blob/master/images/epoch9-10.JPG)

### Epoch 98-100
![alt txt](https://github.com/kshitij1210/Election-Prediction/blob/master/images/epoch98-100.JPG)
