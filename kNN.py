import numpy as np
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, axis=None, val=[], left=None, right=None):
        self.axis = axis
        self.val = val
        self.left = left
        self.right = right

class KDtree(object):
    def __init__(self, point):
        self.dim = len(point[0])
        self.Kbest = None
        root = self.build_tree(point)

    def fit(self, point, depth=0):
        if point == []:
            return None
        axis = depth % self.dim
        array = [p[axis] for p in point]
        median = self.find_median(0, len(array)-1, array, len(array)//2)
        index = array.index(median)
        return Node(axis = axis,
                    val = point[index],
                    left = self.fit(point[:median], deptj=depth+1),
                    right = self.fit(point[median:], depth=depth+1))

    def find_median(self, l, r, a, index):
        i,j = l,r
        mid = l + (r-l)>>1
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
        if l<j and index < j:
            self.find_median(l, j, a, index)
        elif i<r and i < index:
            self.find_median(i, r, a, index)
        return a[index]

    def find_Knearest(self, point, root=None):
        if root is None:
            root = self.root

        if root.left is not None or root.right is not None:
            if point[root.axis] < root.point[root.axis] and root.left is not None:
                self.find_Knearest(point, root=root.left)
            elif root.right is not None:
                self.find_Knearest(point, root=root.right)

        dis = sum(list(map(lambda x,y : abs(x-y), root.point, point)))
        if self.Kbest is None or dis < self.Kbest[0]:
            self.Kbest = (dis, root.point)

        if abs(root.point[root.axis]-point[root.axis]) < self.Kbest[0]:
            if root.right is not None and point[root.axis] < root.point[root.axis]:
                self.find_Knearest(point, root=root.right)
            elif root.left is not None and point[root.axis] >= root.point[root.axis]:
                self.find_Knearest(point, root=root.left)

        return 0


def gen_data(x1, x2):
    y = np.sin(x1) * 1 / 2 + np.cos(x2) * 1 / 2 + 0.1 * x1
    return y


def load_data():
    x1_train = np.linspace(0, 50, 500)
    x2_train = np.linspace(-10, 10, 500)
    data_train = np.array(
        [[x1, x2, gen_data(x1, x2) + np.random.random(1) - 0.5] for x1, x2 in zip(x1_train, x2_train)])
    x1_test = np.linspace(0, 50, 100) + np.random.random(100) * 0.5
    x2_test = np.linspace(-10, 10, 100) + 0.02 * np.random.random(100)
    data_test = np.array([[x1, x2, gen_data(x1, x2)] for x1, x2 in zip(x1_test, x2_test)])
    return data_train, data_test
def main():
    train, test = load_data()
    x_train, y_train = train[:, :2], train[:, 2]
    x_test, y_test = test[:, :2], test[:, 2]  # 同上，但这里的y没有噪声
    t = KDtree(x_train)

    return 0
if __name__ == "__main__":
    main()
    # t = KDtree()
    # a = [3,2,1,5,4,3,2]
    # s = t.find_median(0, len(a)-1, a, len(a)//2)
    # print(s)