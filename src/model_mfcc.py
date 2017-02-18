from mfcc import FeatureExtractor
from spring_new import SpringDTW
import numpy as np
import matplotlib.pyplot as plt


class AQSearch:
    def __init__(self, c_audio_path):
        """
        Initializes the object with MFCC features of the corpus audio.
        :param c_audio_path: path to corpus audio
        """
        self.FE = FeatureExtractor()

        self.c_mfcc_feature_matrix = self.FE.mfcc(c_audio_path)
        self.q_mfcc_feature_matrix = 0

    @staticmethod
    def plot(matrix):
        """
        Function to plot any matrix in the form of heat wave.
        :param matrix:
        :return:
        """
        fig, ax = plt.subplots()
        ax.matshow(matrix, cmap=plt.cm.OrRd)
        plt.show()

    def search(self, q_audio_path):
        """
        Searches for the query audio in a corpus audio.

        :param q_audio_path: path of the query audio to be searched
        :return: none
        """
        # mfcc for query audio is called during search because we can make use of different query files.
        self.q_mfcc_feature_matrix = self.FE.mfcc(q_audio_path)
        sp = SpringDTW(220, self.q_mfcc_feature_matrix, self.c_mfcc_feature_matrix)
        matrix, matches, start_end_data, paths = sp.perform_dtw()
        matrix = np.flipud(matrix)  # flipped upside down
        self.plot(matrix)


if __name__ == '__main__':
    c_wave_path = '/home/gangeshwark/PycharmProjects/AQSearch/data/my4Hellow.wav'
    q_wave_path = '/home/gangeshwark/PycharmProjects/AQSearch/data/queryHellow.wav'
    AQS = AQSearch(c_audio_path=c_wave_path)
    AQS.search(q_audio_path=q_wave_path)
