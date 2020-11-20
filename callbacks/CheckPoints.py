from tensorflow.keras.callbacks import ModelCheckpoint
import os

def CheckPoints(**kwargs):
    # fname(path string): path to while the model would be saved
    # monitor(string): metrics or actribute to monitor for example validation loss, accuracy
    # save_best_only(boolean): determine if to save the best performing model
    # verbose(integer): the verbose
    
    fname = kwargs.get("fname", os.path.sep.join(["models", "new_model.hdf5"]))
    if not os.path.exists(os.path.sep.join([fname, os.pardir])):
        raise FileNotFoundError(f"directory {os.path.exists(os.path.sep.join([fname, os.pardir]))} not found")
        return ""
    monitor = kwargs.get("monitor", "val_loss")
    save_best_only = kwargs.get("save_best_only", True)
    verbose = kwargs.get("verbose", 1)
    mode = kwargs.get("mode", "min")
    best_model = ModelCheckpoint(fname, monitor=monitor, save_best_only=save_best_only, verbose=verbose, mode=mode)
    return best_model