# LogicalFormulaEmbedder
A collection of graph-based logical formula embedding methods

## Experiment Setup

To set up experiments

   1. sh setup_directories.sh
   2. pip install -r requirements.txt (ensures package requirements are met)
   3. sh setup_datasets.sh (downloads and sets up Holstep dataset)
   4. download the mizar files (https://github.com/JUrban/deepmath/tree/master/nndata)
   5. place mizar files into ./datasets/mizar such that they can be found at ./datasets/mizar/nndata

## Processing Data

To process experiment data run

      python3 process_data.py --dataset <dataset>
   
where \<dataset\> is any of holstep, mizar, mizar_cnf, scitail, or all.

      python3 process_data.py --dataset all

FYI: deleting the data is a nightmare because I blow out every formula pair into its own file. I provided a remove data script that can be run 'sh rm_data.sh', but it still takes forever
  
## Training a Model

To train a model

      python3 train_model.py --dataset <dataset>

where \<dataset\> is the dataset you want to experiment with, e.g. 
      
      python3 train_model.py --dataset mizar

validation results are stored in ./results

To enable cuda, simply use the --device <device> flag, generally cuda makes things wwwaaayyyy faster. There's also flags for setting the number of rounds of updates for the MPNN. The full list of flags can be found by running

      python3 train_model.py --h

The default settings are set to those listed in the paper. If one wants to have a faster network, simply lower the dimensionalities where the dataset specification is handled in train_model.py. Generally, having lower dimensionalities doesn't seem to influence final performance on Mizar (haven't checked with Holstep), it just takes longer to converge to the final performance level.

## Testing a Model

To test a model

      python3 test_model.py --dataset <dataset> --model <model_filename>

where \<model_filename\> is the name of the model in ./models you want to test, e.g. 

      python3 test_model.py --dataset holstep --model holstep_epoch_3_MPNN_SimpleMaxPool

test results are stored in ./results

   
## Publications
* If you use this work in your research, please cite the following paper:

```
@misc{crouse2020improving,
      title={Improving Graph Neural Network Representations of Logical Formulae with Subgraph Pooling}, 
      author={Maxwell Crouse and Ibrahim Abdelaziz and Cristina Cornelio and Veronika Thost and Lingfei Wu and Kenneth Forbus and Achille Fokoue},
      year={2020},
      eprint={1911.06904},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
} 
```

  
## Questions
For any question, please contact us via email: Maxwell.Crouse@ibm.com and ibrahim.abdelaziz1@ibm.com

  
  


