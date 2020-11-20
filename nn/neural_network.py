from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import image_data_format as k
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.layers import BatchNormalization as BN

CLASSES = 10
DROPOUT1 = 0.25
DROPOUT2 = 0.5

KERNEL_SIZE= (3,3)
POOL_SIZE = (2,2)

PADDING = "same"
ACTIVATION = "relu"
FINAL_ACTIVATION = "sigmoid"


class VGG16:
    def __init__(self, width : int, height : int, dept : int):
        """[summary]

        Args:
            width ([INTEGER]): width of the input data,
            height ([INTEGER]): height of the input data,
            dept ([INTEGER]): number of dept of the input data (e.g, RBG => 3)
        """
        self.width = width
        self.height = height
        self.dept = dept
            

    def build(self, classes=CLASSES):
        INPUT_SHAPE = (self.height, self.width, self.dept)
        channel_dimension = -1

        if k == "channels_first":
            INPUT_SHAPE = (self.dept, self.height, self.width)
            channel_dimension = 1
        if classes > 2:
            FINAL_ACTIVATION = "softmax"
        model = Sequential()
        # first layer
        model.add(Conv2D(32, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION, input_shape=INPUT_SHAPE))
        model.add(BN(axis=channel_dimension))

        # second layer
        model.add(Conv2D(32, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION))
        model.add(BN(axis=channel_dimension))
        model.add(MaxPool2D(pool_size=POOL_SIZE, padding=PADDING))
        model.add(Dropout(rate=DROPOUT1))

        # third layer
        model.add(Conv2D(64, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION))
        model.add(BN(axis=channel_dimension))

        # fourth layer
        model.add(Conv2D(64, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION))
        model.add(BN(axis=channel_dimension))
        model.add(MaxPool2D(pool_size=POOL_SIZE))
        model.add(Dropout(rate=DROPOUT1))

        # last layer
        model.add(Flatten())
        model.add(Dense(units=500, activation=ACTIVATION))
        model.add(BN(axis=channel_dimension))
        model.add(Dropout(rate=DROPOUT2))
        model.add(Dense(units=classes, activation=FINAL_ACTIVATION))
        
        return model