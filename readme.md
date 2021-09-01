## INSTALLATION
Install dependencies: pip install -r "requirements.txt"


## INPUT LOW LEVEL
A random individual which is taken from the upper level: routes of MC, approximate charging times at each node, E_move of MC, approximate fitness
Note: Using the same low input data to test the performance of the lower level 

In low_input folder, each dataset has a corresponding input folder for low level
- npy folder to be used for python project 
- txt folder to be used for other projects
    + routes (separators: "\t")
    + approximate charging times (separators: "\t")
    + E_move of MC
    + Appoximate fitness (f)

## RUN
Note: Now only the lower level run is possible

Go to main.py, config the parameters
    - Config the chosen file in the f_inputs
    - Config the u_params of the upper level (ACO)
        + Create sample: if not None, always create Npy data. If == "txt", accordingly create "txt" data
    - Config the l_params of the lower level (GWO)

    - Export modes:
        + analysis: return the xlsx file that contain the statistic about the fitness value of optimizer in each run of each file and its average running time
        + details: return the xlsx file in result/data_set folder, which contains the details of fitness value each iteration of each run time of a sepecific file
        + log: return the log file in result/data_set folder, which returns the detail of the the best solution: energy level at each node, charging time, energy remain, energy at depot, energy after being charge, node's status, ...

if run_type['level'] == "upper":
    This will create input low level for the given chosen files
elif run_type['level'] == "lower":
    Optimize the given chosen files using low input datas