## SCA Model Training Repository

### How to use
1. Configure the following environment variables:
    - `WANDB_API_KEY`: WanDB API key for logging. (Optional if wandb logging is turned off in the config)
    - `GITHUB_TOKEN`: GitHub token for accessing private repositories. (Optional)

2. Run the following script to set up the environment. The script will automatically download proper dependencies based on your system configuration. (Windows not supported)
```bash
./configure.sh
```

3. Run the following script to preload the dataset.
```bash
./scripts/preload_dataset.py # Use preload_duplex_dataset.py for full-duplex dataset
```

4. Run the training script. You can modify the configuration file variables as needed.
```bash
./scripts/train.sh # Use train_duplex.sh for full-duplex model training
```

### Configuration files
#### Accelerate configs
preconfigured accelerate config files that use FSDP version 1 is available for 2 and 4 GPUs at `./configs/accelerate/`

#### Training configs
preconfigured training config files are available at `./configs/sca/`
See `./src/sca_train/config/*.py` for more details on configuration options.
