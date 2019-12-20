import heapq
import numpy as np
import matplotlib.pyplot as plt
import sys

class Node(object):
    def __init__(self, axis=None, val=[], left=None, right=None):
        self.axis = axis
        self.point = val
        self.left = left
        self.right = right

class KDtree(object):
    def __init__(self, point, k):
        self.dim = len(point[0])
        self.k = k
        self.Kbest = []
        self.root = self.fit(point)

    def fit(self, point, depth=0):
        if len(point) <= 0 or len(point[0]) == 0:
            return None
        axis = depth % self.dim
        array = [p[axis] for p in point]
        # print(len(point))
        median = self.find_median(0, len(array)-1, array, len(array)//2)
        median = median[len(array)//2]
        index = array.index(median)
        left = [i for i in point if i[axis] < median]
        right = [i for i in point if i[axis] > median]
        return Node(axis = axis,
                    val = point[index],
                    left = self.fit(point=left, depth=depth+1),
                    right = self.fit(point=right, depth=depth+1))

    def find_median(self, l, r, a, index):
        i,j = l,r
        mid = l + (r-l)//2
        x = a[mid]
        while (i<=j):
            while a[i] < x:
                i+=1
            while x < a[j]:
                j-=1
            if i<=j:
                a[i],a[j] = a[j],a[i]
                i +=1
                j -=1
        if l<j and index <= j:
            a = self.find_median(l, j, a, index)
        elif i<r and i <= index:
            a = self.find_median(i, r, a, index)
        return a

    def find_Knearest(self, point, root=None):
        if root is None:
            self.Kbest = []
            root = self.root

        if root.left is not None or root.right is not None:
            if point[root.axis] < root.point[root.axis] and root.left is not None:
                self.find_Knearest(point, root=root.left)
            elif root.right is not None:
                self.find_Knearest(point, root=root.right)

        dis = sum(list(map(lambda x,y : abs(x-y), root.point, point)))
        if len(self.Kbest) < self.k:
            heapq.heappush(self.Kbest,(dis, root.point))
        # print(self.Kbest)
        # print(heapq.nlargest(1, self.Kbest, lambda d:d[0])[0][0])
        # print()
        if dis < heapq.nlargest(1, self.Kbest, lambda d:d[0])[0][0]:
            self.Kbest.pop(-1)
            heapq.heappush(self.Kbest, (dis, root.point))

        # if len(self.Kbest) == 0 or dis < self.Kbest[0]:
        #     self.Kbest = (dis, point)

        if abs(root.point[root.axis]-point[root.axis]) < heapq.nlargest(1, self.Kbest, lambda d:d[0])[0][0]:
            if root.right is not None and point[root.axis] < root.point[root.axis]:
                self.find_Knearest(point, root=root.right)
            elif root.left is not None and point[root.axis] >= root.point[root.axis]:
                self.find_Knearest(point, root=root.left)
        return self.Kbest


def gen_data(x1, x2):
    y = np.sin(x1) * 1 / 2 + np.cos(x2) * 1 / 2 + 0.1 * x1
    return y


def load_data():
    x1_train = np.linspace(0, 50, 500)
    x2_train = np.linspace(-10, 10, 500)
    data_train = [[x1, x2, gen_data(x1, x2) + np.random.random(1)[0] - 0.5] for x1, x2 in zip(x1_train, x2_train)]
    x1_test = np.linspace(0, 50, 100) + np.random.random(100) * 0.5
    x2_test = np.linspace(-10, 10, 100) + 0.02 * np.random.random(100)
    data_test = [[x1, x2, gen_data(x1, x2)] for x1, x2 in zip(x1_test, x2_test)]
    return np.array(data_train), np.array(data_test)

def main():
    train, test = load_data()
    x_train, y_train = train[:, :], train[:, 2]
    # x_test, y_test = test[:, :2], test[:, 2]  # 同上，但这里的y没有噪声
    t = KDtree(x_train, k=3)
    # for i in x_train:
    #     print(t.find_Knearest(i)[-1])
    #     print()
    result = [t.find_Knearest(i)[-1][-1] for i in x_train]
    result = [i[-1] for i in result]
    result = [np.average(i) for i in result]
    #
    print(result)
    plt.plot(result)
    plt.plot(y_train)
    plt.show()


    return 0
if __name__ == "__main__":
    main()
    # t = KDtree([[]],1)
    # a = [3,2,1,5,4,3,2] # 1,2,2,3,3,4,5
    # a = [4,5,6,3,6,8,4] # 3,4,4,5,6,6,8
    # a = [0,2,1,1,5,6,7] # 0,1,1,2,5,6,7
    # s = t.find_median(0, len(a)-1, a, len(a)//2)
    # s = s[len(a)//2]
    # print(s)