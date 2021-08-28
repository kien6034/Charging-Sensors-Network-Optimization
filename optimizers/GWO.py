import numpy as np
import sys
from numpy.lib.function_base import diff
from numpy.random.mtrand import random
from .Parameter import E_MAX, E_MC, E_MIN, INFINITY, MAX_ITER_LOW, NUM_OF_WOLFS, P_M, U, ALPHA
import logging, time
from os import path, mkdir
import matplotlib.pyplot as plt


class GWO:
    def __init__(self, graph, routes, cweights, emove) -> None:
        self.graph = graph
        self.routes = routes[1:-1]
        self.emove =emove
        self.init_cweights= cweights[1:-1]
        self.e_charge = (E_MC - self.emove) 
        self.travel_times = self.get_travel_times(routes)
        #self.generate_logger(self.graph.fileName)
        self.nodes_data = self.get_node_data(self.routes)
    
    # def generate_logger(self, inputFileDir):
    #     inputFileName = inputFileDir.split("/")[-1].replace('.txt', '')

    #     if not path.exists(f'result/{inputFileName}'):
    #         mkdir(f'result/{inputFileName}')
        
    #     fileLogName = f'result/{inputFileName}/{inputFileName}.log'

    #     formatter = logging.Formatter('%(message)s')
    #     file_handler = logging.FileHandler(fileLogName)
    #     file_handler.setFormatter(formatter)

    #     logger.addHandler(file_handler)
       
    
    def get_travel_times(self, routes):
        travel_times = np.zeros((routes.size - 2), dtype=np.float64)
        #param routes start from 0, go over the charging nodes and and at 0
        #self.routes is the set of charging nodes only 
        for i in range(0, routes.size-2):
            travel_times[i] = self.graph.get_tmove(routes[i], routes[i+1])
        
        return travel_times

    def get_node_data(self,routes):
        node_data = np.zeros((routes.size, 2))

        for i in range(routes.size):
            node_data[i, 0] = self.graph.nodes[routes[i], 2]
            node_data[i, 1] = self.graph.nodes[routes[i], 3]
        return node_data

    def init_pops(self, c_weights, l_pop_size, dim):
        pops = np.zeros((l_pop_size, dim))
        for i in range(dim):
            pops[:, i]= np.random.uniform(0, 1, l_pop_size)
        
        pops[0, :] = c_weights
        return pops
        
    
    def fitness(self, idv, logger = None): 
        num_deaths= 0 #max number of death sensor
        max_er = 0 # max energy reduction

        
        t_charge = idv * (self.e_charge / U)

        T = t_charge.sum() + self.emove / P_M #time to finish charging cycle

        if logger:
            logger.info(f"routes: {self.routes}")
            logger.info(f"t_charge: {t_charge}")  
            logger.info(f"T: {T}, energy used: {t_charge.sum() * U + self.emove}")
            logger.info("=================================")     

        t = 0 # time counter
        for i in range(idv.size):
            current_node = self.routes[i]
            t += self.travel_times[i] #time move to node self.routes[i]

            if logger:
                logger.info(f"Node ~{current_node} ~")
                logger.info(f"Init energy: {self.nodes_data[i, 1]}, Charge rate: {self.nodes_data[i, 0]}")
                logger.info(f"Time arrive: {t}, time_travel to {self.routes[i]}: {self.travel_times[i]}")
                

            e_remain = self.graph.nodes[current_node, 3] - t * self.graph.nodes[current_node, 2] #eq3
            e_depot = self.graph.nodes[current_node,3] - T * self.graph.nodes[current_node,2] + t_charge[i] * U #eq 5
            t += t_charge[i]
            
            is_death = False
            if e_remain < E_MIN or e_depot < E_MIN:
                num_deaths += 1
                er = 0
                
                is_death = True
            else:
                diff = self.graph.nodes[current_node, 3] - e_depot
                if diff > 0: #einit > edpot
                    er = diff 
                else:
                    er = 0
            
            if er > max_er:
                max_er = er

            if logger:
                logger.info(f"t_charge: {t_charge[i]}")
                logger.info(f"e_remain: {e_remain}, e_depot: {e_depot}, er: {er}")
                logger.info(f"Energy after charge: {e_remain + t_charge[i] * (U - self.nodes_data[i, 0])}")
                logger.info("-----------------------")
                if is_death:
                    logger.info("!!!!!!!!!!!! DEATH !!!!!!!!!!!!!!")
  
        #fitness 
        f = ALPHA * num_deaths / idv.size + (1- ALPHA) * max_er / (E_MAX - E_MIN)

        if logger:
            logger.info("==============================================")
            logger.info(f"Fitness: {f}, num_deaths: {num_deaths}, max_er: {max_er}")
            logger.handlers.pop()
        
        return f, num_deaths, max_er
      

    def individual_adjustment(self, idv):
        #boundary check
        t = 0 #time counter

        for i in range(idv.size):
            t += self.travel_times[i] #time move to node self.routes[i]
            e_remain = self.graph.nodes[self.routes[i], 3] - t * self.graph.nodes[self.routes[i], 2] #eq3: eint - t * pi

            ub = (E_MAX - e_remain) / (U - self.graph.nodes[self.routes[i], 2]) *  U / self.e_charge #upper bound of idv: max_t_time / total_charge_time

            if idv[i] < 0 or e_remain < E_MIN:
                idv[i] = 0
            elif idv[i] > ub:
                idv[i] = ub
        
            t += idv[i] * self.e_charge/ U #time to charge node self.routes[i]
    
        #if total xi exceed 1
        if idv.sum() > 1:
            idv /= idv.sum()

        return idv

    

    def fix_idv2(self, idv, dies_at, max_er_at,e, current_f):
        #NOTE: some nodes have charging time equal to zeros becuz it is death already (in indivudal adjustment). Try to mingle with that later
        betters_idv = []
        better_weights = []

        for death in dies_at:
            index = death[0]   

            if self.nodes_data[index, 1] <E_MIN: # if the node init energy is lower than E INIT -> death already
                continue

            t_charge = idv * self.e_charge / U

            if death[1] == 1:
                '''To increase eremain, the only way is to make the mc arrive to the current node earlier'''
                #calculate diff time needed
                num_reduced_nodes = index 
                flag = False
                if max_er_at in range(0, index):
                    num_reduced_nodes -= 1
                    flag = True

                if num_reduced_nodes == 0:
                    continue

                diff_energy = E_MIN - e[index, 0] #EMIN - e_remain
                diff_time =  diff_energy / self.nodes_data[index, 0] #amount of time that MC has to arrive to the "current node" earlier: diff_e / pi

                #TODO: not reduced charging time at max u node 
                rn_time = diff_time / num_reduced_nodes #time to be reduced at each node 
                
                #reduce charging time the nodes before the current node
                t_charge[0:index] -= rn_time

                if flag:
                    t_charge[max_er_at] += rn_time
            
            elif death[1] == 2: #dies because of edepot
                #caculate diff energy needed for node to survive when uav arrives depot 
                diff_energy = E_MIN - e[index, 1]  #EMIN - e_depot

                '''increase charging time at that node if possible '''
                #amount of energy that could be charged more
                chargable_energy = (E_MAX - (e[index, 0] + t_charge[index] * (U - self.nodes_data[index,0])))  # E_MAX - (e_remain + t_Charge * (U - pi))

                if chargable_energy >= diff_energy:
                    #increase the charging time at node t
                    t_charge[index] += chargable_energy / (U - self.nodes_data[index, 0])  # to-be-charged T = to-be-charged E / (U - pi)
                else:
                    num_reduced_nodes = idv.size  - (index + 1)
                    flag = 0
                    if max_er_at in range(index+1, idv.size):
                        num_reduced_nodes -= 1
                        flag = 1

                    if num_reduced_nodes == 0:
                        continue

                    #increase the chargable energy at node t
                    t_charge[index] += chargable_energy / (U - self.nodes_data[index, 0])

                    #amount of reduced time of MC spending on charging the after nodes 
                    ttr = (diff_energy - chargable_energy) / self.nodes_data[index, 0]  # timetoreduced = energy / pi
                    ttrn = ttr / num_reduced_nodes

                    t_charge[index+1: idv.size] -= ttrn 
                    if flag == 1:
                        t_charge[max_er_at] += ttrn


            elif death[1] == 3:
                #calculate diff time needed
                num_reduced_nodes = index 
                flag = False
                if max_er_at in range(0, index):
                    num_reduced_nodes -= 1
                    flag = True

                if num_reduced_nodes == 0:
                    continue

                diff_energy = E_MIN - e[index, 0] #EMIN - e_remain
                diff_time =  diff_energy / self.nodes_data[index, 0] #amount of time that MC has to arrive to the "current node" earlier: diff_e / pi

                #TODO: not reduced charging time at max u node 
                rn_time = diff_time / num_reduced_nodes #time to be reduced at each node 
                
                #reduce charging time the nodes before the current node
                t_charge[0:index] -= rn_time

                if flag:
                    t_charge[max_er_at] += rn_time
            
            '''Create new gen and calculate its fitness'''
            #get new gen
            new_idv = t_charge / (self.e_charge / U)
            if new_idv.sum() > 1:
                new_idv /= new_idv.sum()

            n_idv = self.individual_adjustment(new_idv)
            #fitnes
            f, num_deaths, max_er = self.fitness(idv = n_idv)
            
            if f < current_f:   
                betters_idv.append(new_idv)
                better_weights.append((f, num_deaths, max_er))
        return betters_idv, better_weights

    def local_search(self, leaders, l_pop_size):
        new_candidates = []
        new_weights = []
        for leader in leaders:
            num_deaths= 0 #max number of death sensor
            max_er = 0 # max energy reduction
            max_er_at = -1

            t_charge = leader * (self.e_charge / U)
            T = t_charge.sum() + self.emove / P_M #time to finish charging cycle
        
            t = 0 # time counter
            
            dies_at = []
            e = np.zeros((leader.size, 2))
            for i in range(leader.size):
                current_node = self.routes[i]
                t += self.travel_times[i] #time move to node self.routes[i]

                e[i, 0] = self.graph.nodes[current_node, 3] - t * self.graph.nodes[current_node, 2] #eq3
                e[i, 1] = self.graph.nodes[current_node,3] - T * self.graph.nodes[current_node,2] + t_charge[i] * U #eq 5
                t += t_charge[i]
      
                if e[i,0] < E_MIN or e[i, 1] < E_MIN:
                    num_deaths += 1

                    dies_because = 0 #
                    if e[i,0] < E_MIN:
                        dies_because += 1 #
                    if e[i, 1] < E_MIN:
                        dies_because += 2 # 

                    dies_at.append((i, dies_because))
                    er = 0
                else:
                    diff = self.graph.nodes[current_node, 3] - e[i, 1]
                    if diff > 0: #einit > edpot
                        er = diff 
                    else:
                        er = 0
                    
                if er > max_er:
                    max_er = er
                    max_er_at = i
            
            f = ALPHA * num_deaths / leader.size + (1- ALPHA) * max_er / (E_MAX - E_MIN)
            #self.fix_idv(leader, dies_at, e, max_er_at, f)
            better_idvs, better_weights = self.fix_idv2(leader, dies_at, max_er_at,e, f)
            new_weights += better_weights
            new_candidates += better_idvs

            if len(new_candidates) > l_pop_size:
                continue

        return new_candidates, new_weights

    def run(self, l_params):
        #run config
        try:
            l_pop_size = l_params['pop_size']
            l_max_iter = l_params['max_iter']
        except: 
            print("Please define lower level params!")
            sys.exit()
        
        ###################################33
        dim = self.routes.size
        #init population
        pops = self.init_pops(self.init_cweights,l_pop_size, dim)
        weights = np.zeros((l_pop_size, 4), dtype=object)

        #init alpha, beta, delta
        leaders = np.zeros((3, dim))
        leaders_score = np.full((3, 3), float("inf"), dtype=object) #f, num_deaths, max_er

        fs = np.zeros(l_max_iter)

        start_time = time.time()
        #define LB and UB
        for iter in range(0, l_max_iter):
            for i in range(0, l_pop_size):
                if weights[i, 0] == 0:
                    idv = self.individual_adjustment(pops[i])
                    f, num_deaths, max_er = self.fitness(idv = idv)
                else:
                    idv = pops[i]
                    f = weights[i, 1]
                    num_deaths = weights[i, 2]
                    max_er = weights[i, 3]

                #get the index of the worst wolf (highest f)
                worst_idx = np.where(leaders_score[:, 0] == np.amax(leaders_score[:, 0]))[0][0]
    
                if f < leaders_score[worst_idx, 0]:
                    leaders_score[worst_idx] = [f, int(num_deaths), max_er]
                    leaders[worst_idx] = idv.copy()

            a = 2 - iter * (2 / l_max_iter)

            
            # for i in range(0, l_pop_size):
            #     r = np.random.rand(3, dim, 2) #random r1, r2 for each leader and its corresponding dim
            #     A_C  = np.zeros((3, dim, 2))  #A and C
            #     A_C[: ,:, 0] = 2 * a * r[:,:,0] - a #A
            #     A_C[:,:, 1] = 2 * r[:,:,1] #C 

            #     Ds = np.absolute(np.multiply(A_C[:,:,1], leaders) - pops[i])
            #     Xs = leaders - np.multiply(A_C[:,:,0], Ds)
            #     new_pos = np.sum(Xs, axis=0) / 3
            #     pops[i] = new_pos

            #     weights[i, :] = [0, 0, 0, 0]
            
            r = np.random.rand(l_pop_size,3, dim, 2) # 3 x 100 x 100 x 2
            A_C = np.zeros((l_pop_size,3, dim, 2))
            A_C[:,:,:, 0] = 2 * a* r[:,:,:,0] - a
            A_C[:,:,:, 1] = 2 * r[:,:,:, 1]

            Ds = np.absolute(A_C[:,:,:,1] * leaders - pops[:, None, :])
            Xs = leaders - np.multiply(A_C[:,:,:,0], Ds)
            pops = np.sum(Xs, axis = 1) / 3
            weights.fill(0)
        

            best_idx = np.where(leaders_score[:, 0] == np.amin(leaders_score[:, 0]))[0][0]
            fs[iter] = leaders_score[best_idx, 0]

            
            #local search on best idvs
            new_candidates, new_weights = self.local_search(leaders, l_pop_size)
            if len(new_candidates) < l_pop_size:
                for i in range(len(new_candidates)):
                    pops[i, :] = new_candidates[i]
                    weights[i, :] = [1, new_weights[i][0], new_weights[i][1], new_weights[i][2]]
            elif l_pop_size < len(new_candidates):
                for i in range(l_pop_size):
                    pops[i, :] = new_candidates[i]
                    weights[i, :] = [1, new_weights[i][0], new_weights[i][1], new_weights[i][2]]

        best_idx = np.where(leaders_score[:, 0] == np.amin(leaders_score[:, 0]))[0][0]
    
        end_time = time.time()
        run_time = end_time - start_time
        # print(f"Run time was : {run_time}")
        # plt.plot(x, y)
        # plt.show()
                    
        return leaders[best_idx], leaders_score[best_idx, 0], leaders_score[best_idx, 1], leaders_score[best_idx, 2], fs, run_time
          
            