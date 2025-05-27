# Fourier Neural Operator with Laplacian Eigenvectors for Hemodynamic Parameter Upsampling in Aneurysm MRI

## Hyperparameters

Model Parameters used in experiments.

### EDSR Layer

| Parameter | Value |
| --------------- | -----:|
| Hidden Channels |     4 |
| Residual Blocks |     1 |
| Kernel Size     | 3x3x3 |

### Fourier Layer

| Parameter | Value |
| ------------- | -----:|
| Layers        |     4 |
| Fourier Modes | 4x4x4 |
| Channels      |    32 |
| Implicit      |  True |

### Lifting/Projection Layer

| Parameter     | Value L/P |
| ------------- | ---------:|
| Layers        |       2/2 |
| Channels      |   256/128 |

## Ablation

Ablation study for the different hyperparameters in the fourier layer.

| Layers | Hidden Channels | Fourier Modes | *L*    |
| ------:| ---------------:| -------------:| ------:|
|      2 |              16 |             4 | 0.0416 |
|      4 |              16 |             4 | 0.0413 |
|      6 |              16 |             4 | 0.0415 |
|      2 |              16 |             2 | 0.0417 |
|      2 |              16 |             6 | 0.0475 |
|      2 |              32 |             4 | 0.0410 |
|      2 |              64 |             4 | 0.0514 |

## Installation

Clone repository and install requirements:
```shell
git clone https://github.com/moritz-halter/deepflow
cd deepflow
pip install -r requirements.txt
```

## Usage

See [example scripts](https://github.com/moritz-halter/deepflow/blob/master/scripts)

### Preprocessing

Preprocessing should be done in this order:

1. [Grid](#grid)
2. [Surface](#surface)
3. [Geometric prior](#geometric-prior)
4. [Resample](#resample)
5. [Compile](#compile)

#### Grid

To create a grid with to resample the dataset [create_mesh.py](https://github.com/moritz-halter/deepflow/blob/master/scripts/create_mesh.py) can be used:
```shell
[deepflow]$ PYTHONPATH=src python scripts/create_mesh.py --help
usage: create_mesh.py [-h] [-r RESOLUTION] [-t {BOX,SPHRERE,CYLINDER}] -s
                      SAVE_PATH

options:
  -h, --help            show this help message and exit
  -r RESOLUTION, --resolution RESOLUTION
                        Resolution of the mesh in each dimension
  -t {BOX,SPHRERE,CYLINDER}, --type {BOX,SPHRERE,CYLINDER}
                        Type of mesh
  -s SAVE_PATH, --save_path SAVE_PATH
                        save location (if nothing is specified the current
                        working directory is used)
```

#### Surface

[extract_surface.py](https://github.com/moritz-halter/deepflow/blob/master/scripts/extract_surface.py) can be used to extract the surface patch, defined by a saved state (.psvm) produced by [paraview](https://www.paraview.org) where the region of interest was seperated from the rest with a spherical clip function.
```shell
[deepflow]$ PYTHONPATH=src python scripts/extract_surface.py --help
usage: extract_surface.py [-h] -s SOURCE_PATH -p PARAMETER_PATH -o OUTPUT_PATH

extract the surface of a sample region and save it

options:
  -h, --help            show this help message and exit
  -s SOURCE_PATH, --source-path SOURCE_PATH
                        path to the source files (directory containing C0001.vtk)
  -p PARAMETER_PATH, --parameter-path PARAMETER_PATH
                        path to the sample parameter files (directory containing C0001.pvsm)
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        path to save new surface geometry files
```

#### Geometric prior

To generate the geometric priors use [geometric_priors.py](https://github.com/moritz-halter/deepflow/blob/master/scipts/geometric_priors.py).
```shell
[deepflow]$ PYTHONPATH=src python scripts/geometric_priors.py --help
usage: geometric_priors.py [-h] -s SURFACE_PATH -o OUTPUT_PATH

add geometric priors to a surface file

options:
  -h, --help            show this help message and exit
  -s SURFACE_PATH, --surface-path SURFACE_PATH
                        path to the surface files (directory containing C0001_surface.vtk)
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        path to save new surface files
```

#### Resample

To resample the data on a grid imposed on the region of interes use [create_samples.py](https://github.com/moritz-halter/deepflow/blob/master/scripts/create_samples.py).
```shell
[deepflow]$ PYTHONPATH=src python scripts/create_samples.py --help
usage: create_samples.py [-h] -s SOURCE_PATH -g GEOMETRY_PATH -p PARAMETER_PATH -o OUTPUT_PATH [-m MESH_PATH] [-t0 T_START] [-t1 T_END]

preview the sampling for a single timestep

options:
  -h, --help            show this help message and exit
  -s SOURCE_PATH, --source-path SOURCE_PATH
                        path to the source files (directory containing C0001.vtk)
  -g GEOMETRY_PATH, --geometry-path GEOMETRY_PATH
                        path to the surface geometry file (directory containing C0001_surface.vtk)
  -p PARAMETER_PATH, --parameter-path PARAMETER_PATH
                        path to the sample parameter file (directory containing C0001.pvsm)
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        directory to save new sample files in
  -m MESH_PATH, --mesh-path MESH_PATH
                        path to the mesh file
  -t0 T_START, --t-start T_START
                        first time index
  -t1 T_END, --t-end T_END
                        last time index
```

#### Compile

To compile the dataset use [create_dataset.py](https://github.com/moritz-halter/deepflow/blob/master/scripts/create_samples.py).
```shell
[deepflow]$ PYTHONPATH=src python scripts/create_dataset.py --help
usage: create_dataset.py [-h] -r RESOLUTION -o OUTPUT_PATH

compile a dataset datasets

options:
  -h, --help            show this help message and exit
  -r RESOLUTION, --resolution RESOLUTION
                        the grid resolution of the dataset
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        directory to save the dataset in
```

### Training

For training the [super_resolution.py](https://github.com/moritz-halter/deepflow/blob/master/scripts/super_resolution.py) can be used.

```shell
[deepflow]$ PYTHONPATH=src python scripts/super_resolution.py --help
usage: super_resolution.py [-h] [--n_params_baseline N_PARAMS_BASELINE] [--verbose VERBOSE] [--arch ARCH] [--debug DEBUG] [--save_dir SAVE_DIR]
                           [--srcnn.data_channels SRCNN.DATA_CHANNELS] [--srcnn.out_channels SRCNN.OUT_CHANNELS] [--edsr.data_channels EDSR.DATA_CHANNELS]
                           [--edsr.out_channels EDSR.OUT_CHANNELS] [--fno4d.data_channels FNO4D.DATA_CHANNELS] [--fno4d.out_channels FNO4D.OUT_CHANNELS]
                           [--fno4d.n_modes FNO4D.N_MODES] [--fno4d.hidden_channels FNO4D.HIDDEN_CHANNELS] [--fno4d.projection_channels FNO4D.PROJECTION_CHANNELS]
                           [--fno4d.n_layers FNO4D.N_LAYERS] [--fno4d.super_resolution_layer FNO4D.SUPER_RESOLUTION_LAYER]
                           [--fno4d.n_super_resolution_layers FNO4D.N_SUPER_RESOLUTION_LAYERS] [--fno4d.implicit FNO4D.IMPLICIT]
                           [--fno4d.domain_agnostic FNO4D.DOMAIN_AGNOSTIC] [--opt.n_epochs OPT.N_EPOCHS] [--opt.learning_rate OPT.LEARNING_RATE]
                           [--opt.training_loss OPT.TRAINING_LOSS] [--opt.weight_decay OPT.WEIGHT_DECAY] [--opt.scheduler_T_max OPT.SCHEDULER_T_MAX]
                           [--opt.scheduler_patience OPT.SCHEDULER_PATIENCE] [--opt.scheduler OPT.SCHEDULER] [--opt.step_size OPT.STEP_SIZE] [--opt.gamma OPT.GAMMA]
                           [--data.folder DATA.FOLDER] [--data.name DATA.NAME] [--data.batch_size DATA.BATCH_SIZE] [--data.train_test_split DATA.TRAIN_TEST_SPLIT]
                           [--data.train_resolution DATA.TRAIN_RESOLUTION] [--data.test_resolutions DATA.TEST_RESOLUTIONS]
                           [--data.test_batch_sizes DATA.TEST_BATCH_SIZES] [--data.geometric_prior_mode DATA.GEOMETRIC_PRIOR_MODE]
                           [--data.geometric_prior DATA.GEOMETRIC_PRIOR] [--data.super_resolution_rate DATA.SUPER_RESOLUTION_RATE]
                           [--data.super_resolution_dim DATA.SUPER_RESOLUTION_DIM] [--data.n_t DATA.N_T] [--data.noise DATA.NOISE] [--wandb.log WANDB.LOG]
                           [--wandb.name WANDB.NAME] [--wandb.group WANDB.GROUP] [--wandb.project WANDB.PROJECT] [--wandb.entity WANDB.ENTITY]
                           [--wandb.sweep WANDB.SWEEP] [--wandb.log_output WANDB.LOG_OUTPUT] [--wandb.eval_interval WANDB.EVAL_INTERVAL] [--config_name CONFIG_NAME]
                           [--config_file CONFIG_FILE]

options:
  -h, --help            show this help message and exit
  --n_params_baseline N_PARAMS_BASELINE
  --verbose VERBOSE
  --arch ARCH
  --debug DEBUG
  --save_dir SAVE_DIR
  --srcnn.data_channels SRCNN.DATA_CHANNELS
  --srcnn.out_channels SRCNN.OUT_CHANNELS
  --edsr.data_channels EDSR.DATA_CHANNELS
  --edsr.out_channels EDSR.OUT_CHANNELS
  --fno4d.data_channels FNO4D.DATA_CHANNELS
  --fno4d.out_channels FNO4D.OUT_CHANNELS
  --fno4d.n_modes FNO4D.N_MODES
  --fno4d.hidden_channels FNO4D.HIDDEN_CHANNELS
  --fno4d.projection_channels FNO4D.PROJECTION_CHANNELS
  --fno4d.n_layers FNO4D.N_LAYERS
  --fno4d.super_resolution_layer FNO4D.SUPER_RESOLUTION_LAYER
  --fno4d.n_super_resolution_layers FNO4D.N_SUPER_RESOLUTION_LAYERS
  --fno4d.implicit FNO4D.IMPLICIT
  --fno4d.domain_agnostic FNO4D.DOMAIN_AGNOSTIC
  --opt.n_epochs OPT.N_EPOCHS
  --opt.learning_rate OPT.LEARNING_RATE
  --opt.training_loss OPT.TRAINING_LOSS
  --opt.weight_decay OPT.WEIGHT_DECAY
  --opt.scheduler_T_max OPT.SCHEDULER_T_MAX
  --opt.scheduler_patience OPT.SCHEDULER_PATIENCE
  --opt.scheduler OPT.SCHEDULER
  --opt.step_size OPT.STEP_SIZE
  --opt.gamma OPT.GAMMA
  --data.folder DATA.FOLDER
  --data.name DATA.NAME
  --data.batch_size DATA.BATCH_SIZE
  --data.train_test_split DATA.TRAIN_TEST_SPLIT
  --data.train_resolution DATA.TRAIN_RESOLUTION
  --data.test_resolutions DATA.TEST_RESOLUTIONS
  --data.test_batch_sizes DATA.TEST_BATCH_SIZES
  --data.geometric_prior_mode DATA.GEOMETRIC_PRIOR_MODE
  --data.geometric_prior DATA.GEOMETRIC_PRIOR
  --data.super_resolution_rate DATA.SUPER_RESOLUTION_RATE
  --data.super_resolution_dim DATA.SUPER_RESOLUTION_DIM
  --data.n_t DATA.N_T
  --data.noise DATA.NOISE
  --wandb.log WANDB.LOG
  --wandb.name WANDB.NAME
  --wandb.group WANDB.GROUP
  --wandb.project WANDB.PROJECT
  --wandb.entity WANDB.ENTITY
  --wandb.sweep WANDB.SWEEP
  --wandb.log_output WANDB.LOG_OUTPUT
  --wandb.eval_interval WANDB.EVAL_INTERVAL
  --config_name CONFIG_NAME
  --config_file CONFIG_FILE
```

parameters can be set from the command line or in the [config file](https://github.com/moritz-halter/deepflow/blob/master/config/aneurisk_config.yaml), where their purpose is described in more details.


### Interpolation

The scripts [create_interpolation.py](https://github.com/moritz-halter/deepflow/blob/master/scripts/create_interpolation.py) and [run_interpolation.py](https://github.com/moritz-halter/deepflow/scripts/run_interpolation.py)/[run_interpolation_time.py](https://github.com/moritz-halter/deepflow/scripts/run_interpolation_time.py) can be used to created an interpolated dataset or just a prediction for the evaluation set of a different run for later evaluation respectively.
The relevant parameters have to be set in the scripts itself.

### Evaluation

To evaluate and create relevant plots for completed runs [eval_and_plot.py](https://github.com/moritz-halter/deepflow/blob/master/scripts/eval_and_plot.py) can be used.
The script parameters have to be set in the script itself.
