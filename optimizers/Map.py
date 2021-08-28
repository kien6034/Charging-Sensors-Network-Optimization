import matplotlib.pyplot as plt 
import numpy as np
import math
from .Parameter import V, P_M

class Map:
    def __init__(self, fileName) -> None:
        self.fileName= fileName
        self.__numNodes = None
        self.__nodes = None
        self.reader()
        self.__init_ranks = self.get_inital_weights()
        self.__em, self.__tm = self.calculate_e_move()
        self.sort_neigbors_nodes()
    
    @property 
    def nodes(self):
        return self.__nodes

    @property
    def numNodes(self):
        return self.__numNodes
    
    @property
    def init_ranks(self):
        return self.__init_ranks

    def get_emove(self, src, des):
        return self.__em[src, des]

    def get_tmove(self, src, des):
        return self.__tm[src, des]

    def reader(self):
        f = open(self.fileName, "r")
        
        data = dict()
        k = 0
        base = f.readline().split()
        data[k] = (float(base[0]), float(base[1]), -1, -1, 0)
        k += 1

        for row in f:
            values = row.split()
            weight = float(values[2]) / float(values[3])
            data[k] = (float(values[0]), float(values[1]), float(values[2]) * 1.5, float(values[3]), weight)
            k += 1

        self.__numNodes = k 
        num_node_params = 5
        self.__nodes = np.zeros((k, num_node_params), dtype=np.float64)

        for j in range(0, num_node_params):
            self.__nodes[0, j] = data[0][j]
        for i in range(1, k):
            for j in range(0, num_node_params):
                self.__nodes[i,j] = data[i][j]
    
    def draw(self):
        #plt.xlim(0, self.nodes[0][0] * 2)
        #plt.ylim(0, self.nodes[0][1] * 2)
        plt.axis('equal')

        k = 0
        for node in self.__nodes:
            if k == 0:
                plt.plot(node[0], node[1], 'kx', markersize = 10)
            else:
                plt.plot(node[0], node[1], 'bo', markersize = 3)
            k +=1
        plt.show()

    def calculate_e_move(self):
        #calculate e move between 2 nodes
        em = np.zeros((self.numNodes, self.numNodes))
        tm = np.zeros((self.numNodes, self.numNodes))
        
        for i in range(0, self.numNodes):
            for j in range(0, self.numNodes):
                distance = math.sqrt((self.nodes[i][0] - self.nodes[j][0]) ** 2 + (self.nodes[i][1] - self.nodes[j][1]) ** 2)
                em[i, j] =  distance / V * P_M
                tm[i,j] = distance / V

        return em, tm

    def get_inital_weights(self):
        weights = np.zeros(self.numNodes)

        total = np.sum(self.nodes[:, 4])
        for i in range(1, self.numNodes):
            weights[i] = self.nodes[i, 4]  / total
            self.nodes[i, 4] = weights[i]
        
        init_ranks = np.argsort(-1* weights)
        return init_ranks[:-1]
    
    def sort_neigbors_nodes(self):
        #print(self.__tm)
        pass