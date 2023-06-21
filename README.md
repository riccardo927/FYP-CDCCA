# FYP - CDCCA

Firstly the project environment has to be set up in the following steps:
* Download and install Python3 on your local machine, if not already present.
* Download and install `pip` and `pip3` if the installed Python version doesn't already include them.
* From the repository install all the Python libraries needed from the `requirements.txt` file. Install *Pytorch* correctly depending on your machine. Please see more at https://pytorch.org/. 

Secondly, in order to simulate a desired experiment, the file `main.py` needs to be executed. The parameters section in the *main* function of this script can be edited if the financial data, epoch numbers, model architecture, or other hyperparameters are wished to be changed. \

The experiments were run on the NVIDIA T4 Tensor Core GPU, however, depending on your local machine's hardware specifications if a GPU is not present the *device* variable in `main.py` can be changed to GPU. Note that some simulations can take a very long time to: configure the CDCCA model, perform the training, validation, and testing stages, and obtain the final canonical correlations and variables with their respective learning curves and visual representations, if only the CPU is employed. Therefore, a GPU is strongly suggested. \

Alternatively, you can download the Jupyter Notebook, which contains all code execution from all the `.py` files. If all the cells in this notebook are executed on Google Colab with a GPU runtime, then this interactive computational environment allows for a faster computation of all the learning curve plots and canonical variables visualization.
