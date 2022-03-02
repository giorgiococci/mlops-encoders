### Env information
Create conda environment from conda_dependencies.yml:
```bash
conda env create -f conda_dependencies.yml
conda activate mlops-encoders
```
Update the conda environment:
```bash
conda deactivate
conda env update -f conda_dependencies.yml
conda activate mlops-encoders
```

## Scenario
### Create a OneHotEncoder in the training phase and use it in the prediction phase
```bash
python train.py
python inference.py
```
The first script creates the OneHotEncoder and saves it in the `encoders` folder.
The second script uses the OneHotEncoder to encode the data during the inference.