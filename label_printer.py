import pickle
import time
import numpy as np


def main():
    FER_2013_EMO_DICT = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    while 1:
        # Its important to use binary mode
        dbfile = open('probs.p', 'rb')
        # source, destination
        proba = pickle.load(dbfile)
        #values = proba*100
        print(proba)
        #dbfile.close()
        time.sleep(0.5)

if __name__ == '__main__':
    main()
