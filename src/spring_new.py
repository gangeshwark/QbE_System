import numpy as np
from scipy import spatial

from src.readArk import read_scp

"""
The threshold for the matching process has to be chosen by the user - yet in reality the choice of threshold is
a non-trivial problem regarding the quality of the matching process
"""


class SpringDTW:
    def __init__(self, eps, Y_query, X_corpus):
        """
        Initializes the object with the required values for performing DTW.

        :param eps: epsilon (threshold) value for the current search
        :param q_feat: feature matrix of the corpus audio
        :param c_feat: feature matrix of the corpus audio
        """
        self.Y_query = Y_query
        self.X_corpus = X_corpus
        self.epsilon = float(eps)
        self.n = len(self.X_corpus)
        self.m = len(self.Y_query)
        self.D_recent = [float("inf")] * self.m
        self.D_now = [0] * self.m
        self.S_recent = [0] * self.m
        self.S_now = [0] * self.m
        self.d_min = float("inf")
        self.T_s = float("inf")
        self.T_e = float("inf")
        self.check = 0

        # output variables
        self.matches = []
        self.start_end_dist = []
        self.top_k_data = []
        self.all_start_end_dist = []

        self.dist_matrix = np.ndarray(shape=(self.Y_query.shape[0], self.X_corpus.shape[0]), dtype=float)
        self.avg_dist_matrix = np.ndarray(shape=(self.Y_query.shape[0], self.X_corpus.shape[0]), dtype=float)
        print "New SPRING"

    # optimize this function
    def find_start_end(self, new_data):
        if len(self.start_end_dist) == 0:
            self.start_end_dist.append(new_data)
            return

        if self.start_end_dist[-1][0] == new_data[0]:
            if self.start_end_dist[-1][2] >= new_data[2]:
                self.start_end_dist.pop()
                self.start_end_dist.append(new_data)

        else:
            self.start_end_dist.append(new_data)

    def perform_dtw(self):
        """
        Starting the DTW operation

        :return: dist_matrix: The distance matrix
        :return: matches: All the matches
        :return: start_end_dist_data: The start and end points
        :return: all_paths: All the paths for every start and end
        """

        print self.n, self.m
        for j in range(self.n):
            x_t = self.X_corpus[j]
            self.accdist_calc(x_t, self.Y_query, self.D_now, self.D_recent, j)
            self.startingpoint_calc(self.m, self.S_recent, self.S_now, self.D_now, self.D_recent, j)
            if self.d_min < self.epsilon:
                for i in xrange(self.m):
                    if self.D_now[i] >= self.d_min or self.S_now[i] > self.T_e:# or i == self.m:
                        self.T_s = self.S_now[self.m - 1]
                        self.T_e = j - 1
                        """print "REPORT: Distance " + str(self.d_min) + " with a starting point of " + str(
                            self.T_s) + " and ending at " + str(self.T_e)"""
                        self.all_start_end_dist.append([self.T_s, self.T_e, self.d_min])
                        self.find_start_end([self.T_s, self.T_e, self.d_min])
                        self.d_min = float("inf")
                        for a in xrange(self.m):
                            if self.S_now[a] <= self.T_e:
                                self.D_now[i] = float("inf")

            if self.D_now[self.m - 1] <= self.epsilon and self.D_now[self.m - 1] < self.d_min:
                self.d_min = self.D_now[self.m - 1]

            # define the recently calculated distance vector as "old" distance
            # substitution
            for i in range(self.m):
                self.D_recent[i] = self.D_now[i]
                self.S_recent[i] = self.S_now[i]
        print "DTW Done!"
        self.top_k_data = self.top_K(k=20)
        # self.dist_matrix = np.flipud(self.dist_matrix)
        # print self.dist_matrix
        '''
        plt.matshow(self.dist_matrix, cmap=plt.cm.RdGy)
        plt.show()

        paths = self.find_all_paths()
        path_xs = []
        path_ys = []
        for path in paths:
            path_x = []
            path_y = []
            for point in path:
                path_x.append(point[0])
                path_y.append(point[1])
            path_xs.append(path_x)
            path_ys.append(path_y)

        plt.matshow(self.dist_matrix, cmap=plt.cm.RdGy)
        for x in xrange(len(path_xs)):
            plt.plot(path_xs[x], path_ys[x])
        plt.show()
        '''
        paths = []
        return self.dist_matrix, self.matches, self.start_end_dist, paths, self.top_k_data

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
                # distance_new[i] = np.linalg.norm(incoming_value - template[i])**2
                # distance_new[i] = abs(incoming_value - template[i]) ** 2
                distance_new[i] = spatial.distance.cosine(incoming_value, template[i])
                self.dist_matrix[i][j] = distance_new[i]

            else:
                distance_new[i] = (spatial.distance.cosine(incoming_value, template[i])) + \
                                   min(distance_new[i - 1],
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
        i = self.dist_matrix.shape[0] - 1
        path = [[j, i]]
        while i > 0 and j > 0:
            if self.dist_matrix[i - 1, j] == min(self.dist_matrix[i - 1, j - 1], self.dist_matrix[i - 1, j],
                                                 self.dist_matrix[i, j - 1]):
                i -= 1
            elif self.dist_matrix[i, j - 1] == min(self.dist_matrix[i - 1, j - 1], self.dist_matrix[i - 1, j],
                                                   self.dist_matrix[i, j - 1]):
                j -= 1
            elif self.dist_matrix[i - 1, j - 1] == min(self.dist_matrix[i - 1, j - 1], self.dist_matrix[i - 1, j],
                                                       self.dist_matrix[i, j - 1]):
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
        for x in self.top_k_data:
            paths.append(self.find_path(x))
        return paths

    def top_K(self, k=20):
        self.start_end_dist.sort(key=lambda x: x[2])
        return self.start_end_dist[:k]


if __name__ == '__main__':
    # q_bn_feature_matrix = np.array([11, 6, 9, 4])
    # c_bn_feature_matrix = np.array([5, 12, 6, 10, 6, 5, 13])
    c_bn_feature_matrix = read_scp('outdir/bnf_database/raw_bnfea_fbank_pitch.1.scp')
    q_bn_feature_matrix = read_scp('outdir/bnf_query/raw_bnfea_fbank_pitch.1.scp')

    eps = 45
    sp = SpringDTW(eps, q_bn_feature_matrix, c_bn_feature_matrix)
    a, b, c, d, e = sp.perform_dtw()
    print len(e)
    for x in e:
        print x
