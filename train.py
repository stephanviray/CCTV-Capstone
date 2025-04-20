import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO, checks, hub
import multiprocessing

# Import these at the top level
checks()
hub.login('fb3ffac3ac1d81472a488faacbaf15bf0889c72717')

model = YOLO('https://hub.ultralytics.com/models/crZy7LaSPYbd6pBBGoFe')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    results = model.train()