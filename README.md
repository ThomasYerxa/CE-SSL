# CE-SSL

Official repository accompanying the Neurips 2024 paper ["Contrastive-Equivariant Self-Supervised Learning
Improves Alignment with Primate Visual Area IT"](https://proceedings.neurips.cc/paper_files/paper/2024/file/ae28c7bc9414ffd8ffd2b3d454e6ef3e-Paper-Conference.pdf)


## Environment
To install dependencies create a conda environment from the provided `environment.yml` file, and install thei project package by running `pip install -e .` in the base directory.
We utilized Pytorch 1.11 for all experiments and [Composer from MosaicML](https://docs.mosaicml.com/projects/composer/en/stable/index.html) for distributed pretraining on ImageNet datasets.

## Datasets
We provide code for pretraining and online linear evaluation on ImageNet-100/1k.
Our dataset implmentations read images from a ZIP archive of standard datasets rather than opening each image file individually. 
This reduces the I/O overhead of dataloading, but requires zipping the datasets before training which can take up to several hours for ImageNet-1k.
To reproduce training on your cluster you will need to modify `CE_SSL/data/datasets{_paired}.py` to point to the appropriate location for ImageNet in your computing environment. 


## Pretraining
The code is setup to run on a SLURM cluster and uses [submitit](https://github.com/facebookincubator/submitit) for job submission.
To pretrain on ImageNet with default settings run the command:  
```
python3 pretrain_paired.py
```

or 

```
python3 pretrain_vanilla.py
```

to run invariant SSL training. 
See the included training scripts for the available command line arguments (i.e. changing the value of `lambda` in paired/equivariant training).

## Trained Model Weights
We provide pretrained checkpoints for all core models referenced in the main text of the paper. 
See `pretrained_checkpoints.ipynb` for more information/example model loading.
