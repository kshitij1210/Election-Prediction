from keras.models import load_model
import numpy as np

class Classifier(object):

    def __init__(self):
        pass


    def predict():

        testfile = "data/test.csv"
        reader = np.genfromtxt(testfile, delimiter=',')
        x_predict = reader[1:,:]

        #load model binary that was built in build_model
        m = load_model('nn_model.h5')

        #predict winners based on the model
        y_predict = m.predict(x_predict, batch_size=100, verbose=1)

        #write the winners to precictions.csv
        file = open("data/president.csv", "w")
        file.write("President\n")
        for x in np.nditer(y_predict):
            if x > 0.5:
                file.write("Donald Trump\n")
            else:
                file.write("Hillary Clinton\n")
