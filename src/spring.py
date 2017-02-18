import numpy as np
import time
from scipy import spatial

from src.readArk import read_scp

"""
The threshold for the matching process has to be chosen by the user - yet in reality the choice of threshold is
a non-trivial problem regarding the quality of the matching process
"""


class SpringDTW:
    def __init__(self, eps, q_feat, c_feat):
        """
        Initializes the object with the required values for performing DTW.

        :param eps: epsilon (threshold) value for the current search
        :param q_feat: feature matrix of the corpus audio
        :param c_feat: feature matrix of the corpus audio
        """
        self.template = q_feat
        self.stream = c_feat
        self.epsilon = float(eps)
        self.n = len(self.template)
        self.D_recent = [float("inf")] * self.n
        self.D_now = [0] * self.n
        self.S_recent = [0] * self.n
        self.S_now = [0] * self.n
        self.d_rep = float("inf")
        self.J_s = float("inf")
        self.J_e = float("inf")
        self.check = 0

        # output variables
        self.matches = []
        self.start_end_dist_data = []
        self.dist_matrix = np.ndarray(shape=(self.template.shape[0], self.stream.shape[0]), dtype=float)
        print "Old SPRING"

    # calculation of accumulated distance for each incoming value
    def accdist_calc(self, incoming_value, template, distance_new, distance_recent, j):
        """
        Calculation of accumulated distance for each incoming value

        :param incoming_value: the incoming value
        :param template: the feature vector of the corpus audio
        :param distance_new:
        :param distance_recent:
        :param j: The index value of the particular value in the stream
        :return: the new distance vector
        """
        # for eculidean distance of 2 vectors, use dist = np.linalg.norm(a-b)
        for i in range(len(template)):
            if i == 0:
                # distance_new[i] = abs(incoming_value - template[i])**2
                # distance_new[i] = np.linalg.norm(incoming_value - template[i])
                distance_new[i] = spatial.distance.cosine(incoming_value, template[i])

                self.dist_matrix[i][j] = distance_new[i]
            else:
                distance_new[i] = spatial.distance.cosine(incoming_value, template[i]) + min(
                    distance_new[i - 1],
                    distance_recent[i],
                    distance_recent[i - 1])
                self.dist_matrix[i][j] = distance_new[i]

        return distance_new

    # deduce starting point for each incoming value
    def startingpoint_calc(self, template_length, starting_point_recent, starting_point_new, distance_new,
                           distance_recent, j):
        """
        Deduce starting point for each incoming value

        :param template_length: length of the feature vector of the corpus audio
        :param starting_point_recent: the previous starting point
        :param starting_point_new: The new starting point
        :param distance_new: new distance vector
        :param distance_recent: the old distance vector
        :param j: The index value of the particular value in the stream
        :return:
        """
        for i in range(template_length):
            if i == 0:
                # here j+1 instead of j, because of the program counting from 0 instead of from 1
                starting_point_new[i] = j
            else:
                if distance_new[i - 1] == min(distance_new[i - 1], distance_recent[i],
                                              distance_recent[i - 1]):
                    starting_point_new[i] = starting_point_new[i - 1]
                elif distance_recent[i] == min(distance_new[i - 1], distance_recent[i],
                                               distance_recent[i - 1]):
                    starting_point_new[i] = starting_point_recent[i]
                elif distance_recent[i - 1] == min(distance_new[i - 1], distance_recent[i],
                                                   distance_recent[i - 1]):
                    starting_point_new[i] = starting_point_recent[i - 1]
        return starting_point_new

    def find_path(self, data):
        """
        Find the path given a start and end points.
        :param data: contains the start and end data
        :return: the path
        """
        j = data[1]
        i = self.dist_matrix.shape[0]
        path = [[j, i]]
        i -= 1
        j -= 1
        while i > 0 and j > 0:
            if self.dist_matrix[i - 1, j] == min(self.dist_matrix[i - 1, j - 1], self.dist_matrix[i - 1, j],
                                                 self.dist_matrix[i, j - 1]):
                i -= 1
            elif self.dist_matrix[i, j - 1] == min(self.dist_matrix[i - 1, j - 1], self.dist_matrix[i - 1, j],
                                                   self.dist_matrix[i, j - 1]):
                j -= 1
            else:
                i -= 1
                j -= 1
            path.append([j, i])
        return path

    def find_all_paths(self):
        """
        Finds paths for all the set of start and end points.
        :return: paths
        """
        paths = []
        for x in self.start_end_dist_data:
            paths.append(self.find_path(x))

        return paths

    def perform_dtw(self):
        """
        Starting the DTW operation

        :return: dist_matrix: The distance matrix
        :return: matches: All the matches
        :return: start_end_dist_data: The start and end points
        :return: all_paths: All the paths for every start and end
        """
        l = len(self.stream)
        for j in range(l):
            x = self.stream[j]
            self.accdist_calc(x, self.template, self.D_now, self.D_recent, j)
            self.startingpoint_calc(self.n, self.S_recent, self.S_now, self.D_now, self.D_recent, j)

            # Report any matching subsequence
            if self.D_now[self.n - 1] <= self.epsilon:
                if self.D_now[self.n - 1] <= self.d_rep:
                    self.d_rep = self.D_now[self.n - 1]
                    self.J_s = self.S_now[self.n - 1]
                    self.J_e = j + 1
                    """print "REPORT: Distance " + str(self.d_rep) + " with a starting point of " + str(
                        self.J_s) + " and ending at " + str(self.J_e)"""
            # Identify optimal sub-sequence
            for i in range(self.n):
                if self.D_now[i] >= self.d_rep or self.S_now[i] > self.J_e:
                    self.check += 1
            if self.check == self.n:
                """
                print "MATCH: Distance " + str(self.d_rep) + " with a starting point of " + str(
                    self.J_s) + " and ending at " + str(self.J_e)
                """
                # self.matches.append(str(self.d_rep) + "," + str(self.J_s) + "," + str(self.J_e))
                self.matches.append([self.J_s, self.J_e, self.d_rep])
                self.find_start_end([self.J_s, self.J_e, self.d_rep])
                self.d_rep = float("inf")
                self.J_s = float("inf")
                self.J_e = float("inf")
                self.check = 0
            else:
                self.check = 0
            # define the recently calculated distance vector as "old" distance
            for i in range(self.n):
                self.D_recent[i] = self.D_now[i]
                self.S_recent[i] = self.S_now[i]
        """for (x, y, z) in self.matches:
            hypo = np.sqrt(np.square(y - x) + np.square(self.dist_matrix.shape[0]))
            norm_dist = z / hypo
            if norm_dist <= float("inf"):
                print norm_dist, x, y
                self.start_end_dist_data.append([x, y, norm_dist])
        """

        print "DTW Done!"
        self.top_k_data = self.top_K(k=20)
        paths = []
        return self.dist_matrix, self.matches, self.start_end_dist_data, paths, self.top_k_data

    def top_K(self, k=20):
        self.start_end_dist_data.sort(key=lambda x: x[2])
        return self.start_end_dist_data[:k]

    # optimize this function
    def find_start_end(self, new_data):
        if len(self.start_end_dist_data) == 0:
            self.start_end_dist_data.append(new_data)
            return

        if self.start_end_dist_data[-1][0] == new_data[0]:
            if self.start_end_dist_data[-1][2] >= new_data[2]:
                self.start_end_dist_data.pop()
                self.start_end_dist_data.append(new_data)

        else:
            self.start_end_dist_data.append(new_data)


if __name__ == '__main__':
    test = 0
    if test:
        q_bn_feature_matrix = np.array([11, 6, 9, 4])
        c_bn_feature_matrix = np.array([5, 12, 6, 10, 6, 5, 13])
    else:
        c_bn_feature_matrix = read_scp('outdir/bnf_database/raw_bnfea_fbank_pitch.1.scp')
        q_bn_feature_matrix = read_scp('outdir/bnf_query/raw_bnfea_fbank_pitch.1.scp')
    eps = 30
    sp = SpringDTW(eps, q_bn_feature_matrix, c_bn_feature_matrix)
    st = time.time()
    a, b, c, d = sp.perform_dtw()
    et = time.time()
    print "%s seconds" % (et-st)
    #print a
