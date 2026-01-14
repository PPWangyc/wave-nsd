# wave-nsd

Whole brain NSD decoding with fMRI-text-image contrastive alignment.

## Setup

Create the conda environment:

```bash
conda env create -f env.yaml
conda activate wave
```

## Data

NSD raw and processed data are available on the sc cluster at:
```
/viscam/projects/neural-decoding/data/NSD
```

To process NSD data, run the notebook:
```bash
jupyter notebook notebooks/create_nsd.ipynb
```

This will generate the processed NSD data needed for training.

## Training

Train the fMRI-text-image contrastive alignment model:

```bash
cd ..
source train_contrastive_nsd.sh
```

## Architecture

The fMRI encoder uses a 3D convolutional layer followed by an MLP backbone:
- 3D Conv3d layer (kernel size 4×4×4, stride 4)
- AdaptiveMaxPool3d to (8, 8, 8)
- 4-layer MLP with LayerNorm and GELU activation

See `src/model/fvl.py:382-406` for implementation details.
