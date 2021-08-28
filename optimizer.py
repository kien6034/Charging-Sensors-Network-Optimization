from optimizers.Map import *
from optimizers.ACO import * 
from optimizers.GWO import *
from os import path, mkdir
import numpy as np
import pandas as pd
import sys
import logging 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_low_data(graph):
    fileDir = graph.fileName
    xs = fileDir.split("/")
    fileName = xs[-1].replace('.txt', '')
    expected_dir = f"low_input/{fileName}"

    if not path.exists(expected_dir):
        print("Data is not created! Please run the upper level first to get the test data")
        sys.exit()
    
    routes = np.load(f"{expected_dir}/routes.npy")
    cweights = np.load(f"{expected_dir}/cweights.npy")
    f = np.load(f"{expected_dir}/f.npy")
    e_move = np.load(f"{expected_dir}/emove.npy")
    num_deaths = np.load(f"{expected_dir}/num_deaths.npy")
    max_er = np.load(f"{expected_dir}/max_er.npy")
    return routes, cweights, f, e_move, num_deaths, max_er

def create_output_folder(inputDir):
    inputFileName = inputDir.split("/")[-1].replace('.txt', '')

    if not path.exists(f'result/{inputFileName}'):
        mkdir(f'result/{inputFileName}')
    
    return f'result/{inputFileName}'

def generate_logger(inputFileDir):
    inputFileName = inputFileDir.split("/")[-1].replace('.txt', '')

    if not path.exists(f'result/{inputFileName}'):
        mkdir(f'result/{inputFileName}')
    
    fileLogName = f'result/{inputFileName}/{inputFileName}.log'

    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(fileLogName)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def run(run_type, f_input, u_params, l_params, export_flags):
    level = run_type['level']
    num_of_runs = run_type['num_of_runs']

    input_url = f_input['input_url']
    num_nodes = f_input['num_nodes']
    distribution_type = f_input['distribution']
    file_nums = f_input['file_nums']
    
    file_urls = list()
    for num_node in num_nodes:
        for file_num in file_nums:
            file_name = str(distribution_type) + str(num_node) + "_0" + str(file_num) + "_simulated.txt"
            file_url = input_url + file_name
            file_urls.append(file_url)

    #export parameter
    exports = {'convergence': False, 'details': False, 'analysis': False, 'log': False}
    for k in export_flags:
        if k in exports:
            exports[k] = export_flags[k]
        else:
            print(f"The export type: {export_flags[k]} - is not supported yet!")
            sys.exit()
    
    #init export data holder
    if exports['analysis']:
        data_analysis = dict()
        data_analysis['data_set'] = list()
        data_analysis['avg_run_time'] = list()
        for i in range(num_of_runs):
            data_analysis[f'run {i+1}'] = list()
    
    #run
    for file_url in file_urls:
        print(f"Optimizing the data: {file_url.split('/')[-1]}")
        logger = generate_logger(file_url)


        if exports['details']:
            data_details = dict()
            data_details['data_set'] = list()
            data_details['run_number'] = list()
            data_details['run_time'] = list()
            data_details['num_deaths'] = list()
            data_details['max_er'] = list()
            if run_type['level'] == "lower":
                for i in range(l_params['max_iter']):
                    data_details[f'iter {i}'] = list()

        graph = Map(file_url)
        out_folder_url = create_output_folder(file_url)
        file_name = file_url.split('/')[-1].replace('_simulated.txt', '')

        if exports['analysis']:
            data_analysis['data_set'].append(file_name)


        #=================================================================================
        total_run_times = 0
        for i in range(num_of_runs):
            if exports['details']:
                data_details['data_set'].append(file_name)
               
            if level == "upper": #run upper
                upper = ACO(graph)
                upper.run(create_sample = True)
            elif level == "lower":
                routes, cweights, f, emove, num_deaths, max_er = read_low_data(graph)
                lower = GWO(graph, routes, cweights, emove)
                idv, f, num_deaths, max_er, f_details, run_time = lower.run(l_params)
                total_run_times += run_time

                if exports['analysis']: 
                    data_analysis[f'run {i+1}'].append(f)
                
                if exports['details']:
                    data_details['num_deaths'].append(num_deaths)
                    data_details['max_er'].append(max_er)
                    data_details['run_number'].append(i)
                    data_details['run_time'].append(run_time)
                    for j in range(f_details.size):
                        data_details[f'iter {j}'].append(f_details[j])
                
                if exports['log']:
                    if i == (num_of_runs - 1):
                        f, num_deaths, max_er = lower.fitness(idv,logger)

            else:   
                print("Level is not defined!")
                sys.exit()

        if exports['analysis'] and run_type['level'] == "lower": 
            data_analysis['avg_run_time'].append(total_run_times/num_of_runs)
            df = pd.DataFrame(data_analysis)            
            df.to_excel(rf'analysis.xlsx', index = False, header=True)

        if exports['details']and run_type['level'] == "lower":
            df = pd.DataFrame(data_details)            
            df.to_excel(f'{out_folder_url}/details.xlsx', index = False, header=True)
