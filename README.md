# Hierarchical Emergence Profiles of Human-Derived Dimensions are a Fundamental Property of Deep Neural Networks

## Overview 

In this repository, you will find all code related to the study: Hierarchical Emergence Profiles of Human-Derived Dimensions are a Fundamental Property of Deep Neural Networks.

## Installation 

*pip*: 
To install all required libraries, you can use the requirements.txt. To install from it, make sure you are in the right folder, and then install using "pip install -r requirements.txt"

*conda*: 
To install the libraries via conda, use the "environment.yml" file. By then activating the created enviroment, you should be able to run all code needed. 

## Datasets 

To repeat the analysis, you first need to download the data for the two dataset we used: 

*THINGS*: A large-scale image dataset of 1,854 everyday object concepts with human-derived behavioral similarity ratings, used to model object-level representations in vision and cognition. You can download it under [https://osf.io/z2784/](https://osf.io/z2784/) or [THINGS](https://things-initiative.org). 

*STUFF*: A curated image dataset of 600 natural and artificial material categories with human-derived perceptual dimensions, designed to study material perception. To acess the data and dimensions, see [https://osf.io/5gr73/.](https://osf.io/5gr73/).   

## Key Files

### Analysis
The main file needed for the analysis is the "running_Analysis.py" file which allows you to run the method for all models, datasets, and settings. It calls the activation extraction functions from the layer_extractions folder which contains one function/file for each model (and pixel extraction). It will also create all plots once the results have been fully created.

**To use DINOv3 you need to request access on Huggingface, see [here](https://huggingface.co/docs/transformers/main/en/model_doc/dinov3)**

### Plotting
The plots are automatically created when running the "running_Analysis.py" file. If you just want to run the plots, you can use the "running_Plots.py" file. All plots can be found in the "Results" folder under "final_plots". 

## Citation 

If you use any code from this repository, please cite: 

ADD CITATION 

## Licence 

This project is licensed under the MIT License – see the LICENSE file for details.

## Contact 

Should you have any questions or trouble running the analysis, please contact me at: F.Burger@westernsydney.edu.au

 
