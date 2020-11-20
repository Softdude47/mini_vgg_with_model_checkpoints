print("[INFO] importing librabries...")
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from callbacks.CheckPoints import CheckPoints
from nn.neural_network import VGG16
import os
print("[INFO] done....")

# loading data
print("[INFO] loading data...")
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
lb = LabelBinarizer()
print("[INFO] done...")

# rearanging the features
xtrain = xtrain.astype("float")/255.0
xtest = xtest.astype("float")/255.0

# one hot encoding
ytrain = lb.fit_transform(ytrain)
ytest = lb.fit_transform(ytest)

# print(f"ytrain: {ytrain.shape}, xtrain: {xtrain.shape} \nytest: {ytest.shape}, xtest: {xtest.shape}")
# variables used in model class parameter(VGG16)
classes = 10
width, height, dept = xtrain.shape[1:]
multiclass = True

# variables used in model training
epochs = 40
batch = 32

# variables used in model compilation
loss = "categorical_crossentropy"
metrics = ["accuracy"]

# variables used in model optimzer
nesterov = True
momentum = 0.9
learning_rate = 0.1

# variables used in model callback
fname = os.path.sep.join([os.getcwd(), "models", "Cifar10Model.hdf5"])
save_best_only = True
verbose = 1
mode = "min"

# callback
callback = CheckPoints(fname=fname, save_best_only=save_best_only, verbose=verbose, mode=mode)
callback_list = [callback]

# model
sgd = SGD(lr=learning_rate, nesterov=nesterov, momentum=momentum) 
model = VGG16(width=width, height=height, dept= dept).build(classes=classes)
model.compile(loss=loss, metrics=metrics, optimizer=sgd)
print("[INFO] trainning model")
model.fit(xtrain, ytrain, validation_data=[xtest, ytest], verbose=1, epochs=epochs, batch_size=batch, callbacks=callback_list)