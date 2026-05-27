import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from moveminerx.utils.util import read_shapefile, load_mat, save_shapefile
from matplotlib import pyplot as plt


def get_distance_matrix(datas):
    n = np.shape(datas)[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v_i = datas[i, :]
            v_j = datas[j, :]
            distance_matrix[i, j] = np.sqrt(np.dot((v_i - v_j), (v_i - v_j)))
    return distance_matrix


def select_dc(distance_matrix):
    n = np.shape(distance_matrix)[0]
    distance_array = np.reshape(distance_matrix, n * n)
    percent = 10.0 / 100
    position = int(n * (n - 1) * percent)
    dc = np.sort(distance_array)[position + n]
    return dc


def get_local_density(distance_matrix, dc, method=None):
    n = np.shape(distance_matrix)[0]
    rhos = np.zeros(n)
    for i in range(n):
        if method is None:
            rhos[i] = np.where(distance_matrix[i, :] < dc)[0].shape[0] - 1
        else:
            pass
    return rhos


def get_deltas(distance_matrix, rhos):
    n = np.shape(distance_matrix)[0]
    deltas = np.zeros(n)
    nearest_neighbor = np.zeros(n)
    rhos_index = np.argsort(-rhos)
    for i, index in enumerate(rhos_index):
        if i == 0:
            continue
        higher_rhos_index = rhos_index[:i]
        deltas[index] = np.min(distance_matrix[index, higher_rhos_index])
        nearest_neighbors_index = np.argmin(distance_matrix[index, higher_rhos_index])
        nearest_neighbor[index] = higher_rhos_index[nearest_neighbors_index].astype(int)
    deltas[rhos_index[0]] = np.max(deltas)
    return deltas, nearest_neighbor


def find_k_centers(rhos, deltas, k):
    rho_and_delta = rhos * deltas
    centers = np.argsort(-rho_and_delta)
    return centers[:k]


def density_peal_cluster(rhos, centers, nearest_neighbor):
    k = np.shape(centers)[0]
    if k == 0:
        print("Can't find any center")
        return
    n = np.shape(rhos)[0]
    labels = -1 * np.ones(n).astype(int)

    for i, center in enumerate(centers):
        labels[center] = i

    rhos_index = np.argsort(-rhos)
    for i, index in enumerate(rhos_index):
        if labels[index] == -1:
            labels[index] = labels[int(nearest_neighbor[index])]
    return labels


def generate_gauss_datas():
    first_group = np.random.normal(20, 1.2, (100, 2))
    second_group = np.random.normal(10, 1.2, (100, 2))
    third_group = np.random.normal(15, 1.2, (100, 2))

    datas = []
    for i in range(100):
        datas.append(first_group[i])
        datas.append(second_group[i])
        datas.append(third_group[i])
    datas = np.array(datas)
    return datas


def draw_decision(datas, rhos, deltas):
    n = np.shape(datas)[0]
    for i in range(n):
        plt.scatter(rhos[i], deltas[i], s=16, color=(0, 0, 0))
        plt.annotate(str(i), xy=(rhos[i], deltas[i]), xytext=(rhos[i], deltas[i]))
        plt.xlabel('local density-ρ')
        plt.ylabel('minimum distance to higher density points-δ')
    plt.show()


class DPC2:
    def __init__(self, r, n):
        self.r = r # 阈值参数
        self.n = n # 超参数

    def fit(self, X):
        # 计算每个数据点的密度
        N = len(X)
        neighbor = NearestNeighbors(n_neighbors=self.n+1).fit(X)
        dist, idx = neighbor.kneighbors(X)
        density = 1. / (dist[:, 1:].sum(axis=1) / self.n)

        # 找到密度峰值点
        order = np.argsort(density)[::-1]
        max_idx = order[0]
        center = [max_idx]
        center_density = [density[max_idx]]
        remain_idx = np.setdiff1d(order, center)
        while remain_idx.size > 0 and density[remain_idx[0]] > 0:
            neighbor = NearestNeighbors(n_neighbors=1).fit(X[center])
            dist, idx = neighbor.kneighbors(X[remain_idx])
            new_center = remain_idx[dist.max(axis=1) > self.r]
            if len(new_center) == 0:
                break
            new_density = density[new_center]
            if np.max(new_density) > density[max_idx]:
                argmax = np.argmax(new_density)
                max_idx = new_center[argmax]
                center.append(max_idx)
                center_density.append(new_density[argmax])
            remain_idx = np.setdiff1d(remain_idx, new_center)

        # 根据簇中心构建聚类簇
        self.labels_ = -1 * np.ones(N)
        for i, c in enumerate(center):
            self.labels_[dist[c] < self.r] = i
        self.n_cluster_ = len(center)
        self.density_ = np.array(center_density)

        return self


def main():

    distance_matrix = load_mat(r'simulated_Ex\TC2_dspd2.mat')['TC2']
    distance_matrix = load_mat(r'simulated_Ex\T1_dspd.mat')['p_dist_2']
    dc = select_dc(distance_matrix)
    rhos = get_local_density(distance_matrix, dc)
    deltas, nearest_neighbor = get_deltas(distance_matrix, rhos)
    centers = find_k_centers(rhos, deltas, 3)
    labels = density_peal_cluster(rhos, centers, nearest_neighbor)
    print(labels)
    # draw_decision(datas, rhos, deltas)
    # plt.cla()
    # fig, ax = plt.subplots()
    # for i in range(300):
    #     if labels[i] == 0:
    #         ax.scatter(datas[i, 0], datas[i, 1], facecolor='C0', edgecolors='k')
    #     elif labels[i] == 1:
    #         ax.scatter(datas[i, 0], datas[i, 1], facecolor='yellow', edgecolors='k')
    #     elif labels[i] == 2:
    #         ax.scatter(datas[i, 0], datas[i, 1], facecolor='red', edgecolors='k')
    # plt.show()


# if __name__ == '__main__':
#     main()
