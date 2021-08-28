import numpy as np
from numpy.random.mtrand import rand, randint
from .Parameter import *
import copy, math
import sys
import matplotlib.pyplot as plt
from os import path, mkdir

#TODO: Upper level
"""
    - Try different pheromones update params: alpha, beta, m, EVAPORATION_RATE, DEPOSIT_RATE
    - Lighten code
    - Apply local search
"""

class ACO:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.pheromones = None
    
    def init_pheromone(self):
        return np.full((self.graph.numNodes, self.graph.numNodes), MIN_PHEROMONE)
    
    def appox_evaluate(self, routes, E_Move):
        appx_cweights = np.zeros(routes.size) 

        t = (E_MC - E_Move) / U
        for i in range(1, routes.size -1):
            appx_cweights[i] = self.graph.nodes[routes[i], 4]
        
        num_deaths= 0 #max number of death sensor
        max_er = 0 # max energy reduction

        #calculate T 
        appx_T = E_Move/P_M + (E_MC - E_Move) / U
        a = 0
  
        for i in np.arange(1, len(routes) -1):
            current_node = routes[i]
            a += self.graph.get_tmove(routes[i-1], routes[i])

            e_remain = self.graph.nodes[current_node, 3] - a * self.graph.nodes[current_node,2] #eq3
            e_depot = self.graph.nodes[current_node,3] - appx_T * self.graph.nodes[current_node, 2] + appx_cweights[i] * t * U

            a += appx_cweights[i] * t
            if e_remain < E_MIN or e_depot < E_MIN: #zi = 1
                num_deaths +=1
                er = 0
            else: #zi = 0
                diff = self.graph.nodes[current_node, 3] - e_depot 
                if diff > 0:  # enit > edpot
                    er = diff 
                else:
                    er = 0
            
            if er > max_er:
                max_er = er
        
        #calculate fintness value
        f = ALPHA * num_deaths / (routes.size - 2) +(1- ALPHA) * max_er / (E_MAX - E_MIN)
        return f, appx_cweights, num_deaths, max_er
      

    def global_pheromone_update(self, best_route):
        for i in range(1, best_route.size):
            self.pheromones[best_route[i-1], best_route[i]] *= (1 + DEPOSIT_RATE)

    def save_low_input(self, routes, cweights, f, E_Move, num_deaths, max_er):
        fileDir = self.graph.fileName

        xs = fileDir.split("/")
        fileName = xs[-1].replace('.txt', '')

        expected_dir = f"low_input/{fileName}"
        if not path.exists(expected_dir):
            mkdir(expected_dir)
        
        
        np.save(f"{expected_dir}/routes", routes)
        np.save(f"{expected_dir}/cweights", cweights)
        np.save(f"{expected_dir}/f", f)
        np.save(f"{expected_dir}/emove", E_Move)
        np.save(f"{expected_dir}/num_deaths", num_deaths)
        np.save(f"{expected_dir}/max_er", max_er)
        print(f"input for {fileName} is created")

    def run(self, create_sample):
        #init pheromone 
        self.pheromones = self.init_pheromone()
        #print(f"ACO pheromone mattrix initialized with the shape of {self.pheromones.shape} and init value {MIN_PHEROMONE}")

        ant = Ant(self.graph)
        route_len = self.graph.nodes.shape[0] + 1

        for iter in range(MAX_ITERATION):
            iterO = INFINITY

            pheromones = copy.deepcopy(self.pheromones)
            best_routes = np.empty((NUM_BEST_ANTS, route_len), dtype=np.int16)
            best_cweights = np.empty((NUM_BEST_ANTS, route_len), dtype=np.float64)
            best_fs = np.full(NUM_BEST_ANTS, INFINITY)

            for ant_iter in range(NUM_OF_ANTS):
                alpha = min(5, 0.04 * (iter + 1))
                beta = min(5, 0.04 * (iter + 1))
                routes, E_Move = ant.find_route2(pheromones, alpha, beta) 
                f, appx_cweights, num_deaths, max_er = self.appox_evaluate(routes, E_Move)

                ant.update_local_pheromone(pheromones, routes)
                
                if create_sample:
                    self.save_low_input(routes, appx_cweights, f, E_Move, num_deaths, max_er)
                    return 0

                #update best fitness
                max_f_idx = np.where(best_fs == np.amax(best_fs))[0][0]
                
                if f < best_fs[max_f_idx]:
                    best_fs[max_f_idx] = f
                    best_routes[max_f_idx] = routes 
                    best_cweights[max_f_idx] = appx_cweights
                #TODO: local search
                #TODO: Find the best route of best ants using GWO
            
            # x.append(iter)
            # y.append(best_fs[0])
            
            print(f"iter {iter}")
            print(best_fs[0])
            print("====================")
            # if iter == MAX_ITERATION - 1:
              
            #     plt.plot(x, y)
            #     plt.show()
            
            
            self.global_pheromone_update(best_routes[0])
                
        

class Ant:
    def __init__(self, graph) -> None:
        self.graph = graph
        pass

    def find_route2(self, pheromones, alpha, beta):
        routes_len = self.graph.nodes.shape[0] + 1
        c_len = routes_len - 2

        C = set(np.arange(1, routes_len - 1))
        routes = list()
        routes.append(0)

        while C:
            current_node = routes[-1]
            m = randint(MIN_NUM_CANDIDATES, routes_len - 1)

            if m > len(C):
                m = len(C)
            
            next_node = self.move(current_node, C, m, pheromones, alpha, beta)
            
            routes.append(next_node)
            C.remove(next_node)
        routes.append(0)

        #get EMove
        E_Move = 0
        for j in range(len(routes) - 1):
            E_Move += self.graph.get_emove(routes[j], routes[j+1])
        
        return np.asarray(routes), E_Move
        

    def move(self,current_node, avail_nodes, m, pheromones, alpha, beta):
        #get candidates
        candidates = np.empty(m, dtype=np.int16)
        candidates_weights = np.full(m, INFINITY)
        
        for neigbor in avail_nodes:
            weight = self.graph.get_tmove(current_node, neigbor) 
            if weight == 0:
                return neigbor
                
            #weight = distance * 1 / (self.graph.nodes[neigbor, 4])
            max_idx = np.where(candidates_weights == np.amax(candidates_weights))[0][0]
            if weight < candidates_weights[max_idx]:
                candidates_weights[max_idx] = weight
                candidates[max_idx] = neigbor            

        total = 0
        for i in range(candidates_weights.size):
            candidates_weights[i] = math.pow(pheromones[current_node, candidates[i]], alpha) * math.pow((1 / candidates_weights[i]), beta) 
            total += candidates_weights[i]

        candidates_weights /= total
        idxs = np.arange(candidates_weights.size)
        next_idx = np.random.choice(idxs, 1, p=candidates_weights)[0]
        return candidates[next_idx]
    
    
    def update_local_pheromone(self, pheromones, routes):
        for i in range(1, routes.size):
            pheromones[routes[i-1], routes[i]] *= (1 - LOCAL_EVAPORATION_RATE)
       
    