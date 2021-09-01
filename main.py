from optimizer import run 

#FOR TEST: choose the level to run
run_type = {
    'level': 'lower',
    'num_of_runs':10
}

#file input
f_input = {
    'input_url': "small-net/grid/base_station_(250.0, 250.0)/",
    'distribution': 'gr',
    'num_nodes': [25, 50, 75, 100], 
    'file_nums': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

#upper params 
u_params = {
    'max_iter': 50,
    'pop_size': 30, 
    'create_sample': "txt"
}

#lower params
l_params = {
    'max_iter': 20,
    'pop_size': 50
}

#export flag 
export_flags = {
    'convergence': True,
    'details': True,
    'analysis': True,
    'log': True
}

#TODO: 
# upper level using ACO 
# plot convergence
# read new article about optimizing GWO

run(run_type, f_input, u_params, l_params, export_flags)