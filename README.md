# Timbre-Qualifier-SMC

**Description**: Code repository for the paper "Timbre Trait Prediction for Performance Analysis of Musical Sound Synthesizer using Deep Neural Embeddings". This repository contains scripts to reproduce the experiments presented in the paper, the training of the timbre model and the assessment of the synthesizer.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/TheoChasleCauchy/timbre-qualifier-SMC.git
   cd timbre-qualifier-SMC
   ```

2. Create a virtual environment (optional)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

2. Install requirements:
    ```bash
    pip install .
    ```

## Dataset
Download the [RWC Dataset](https://zenodo.org/records/17170844) and place it in `./data/RWC/`.

## Configuration
- Experiment parameters (e.g., models architectures, embeddings used, training parameters) are defined in:
  - `./experiments/timbre-model/config.yaml`
  - `./experiments/synthesizer-assessment/config.yaml`
- Edit these files to customize the experiments.

## Experiments

1. Process the RWC samples:
    ```bash
    python ./process_RWC/process_RWC.py
    ```
    This will launch the process pipeline of RWC, split the RWC audio files into single notes samples available in `./data/RWC/RWC_processed/`, then compute the embeddings (CLAP, CLAP Music, VGGish and MERT) and save them in `./data/embeddings/`.
    
    **Pipeline**:
    - Split of the RWC audio files into single notes samples available in `./data/RWC/RWC_processed/`
    - Creation of a metadata CSV file of the RWC samples saved in `./data/RWC/metadata/`
    - Computation of the embeddings (CLAP, CLAP Music, VGGish and MERT) and save them in `./data/RWC/embeddings/`
    - Creation of a metadata CSV file of the RWC embeddings saved in `./data/RWC/metadata/`
    - Creation of a train-valid split (Two distinct CSV files) depending on the split in split_config.yaml.
    
        NB: If you want to change the split, use :
        ```bash
        python ./process_RWC/change_split.py # If you want to specify yourself the split in split_change.yaml
        python ./process_RWC/change_split.py -r # If you want to randomly change thesplit in split_change.yaml
        python ./process_RWC/change_split.py -r --train_proportion # If you want specify the train split proportion (default 0.8)
        python ./process_RWC/change_split.py -r --random_seed # If you want specify the random seed (default 1) 
        ```

2. Launch the timbre-model pipeline to train the models
    ```bash
    python ./experiments/timbre-model/main.py
    ```
    This will launch the training of all the models considering the parameters in `./experiments/timbre-model/config.yaml` and will generate the results files and figures in the `./results` folder.
    
    **Pipeline**:
    - Compute a random train-valid split of the dataset
    - Train each model with a cross-validation approach, saved in `./models/cross-validation/`
    - Compute the cross-validation predictions and metrics, saved in `./results/timbre-model/metrics/`
    - Generation of figures, saved in `./results/timbre-model/figures/`


3. Launch the synthesizer-assessment pipeline assess the synthesizer

    You first need to train a model on all instruments by specifing the parameters in the `./experiments/synthesizer-assessment/config.yaml` file (by default the model is trained on CLAP emneddings with no hidden layers):
    ```bash
    python ./experiments/synthesizer-assessment/train_model.py
    ```
    Then you can launch the pipeline:
    ```bash
    python ./experiments/synthesizer-assessment/main.py
    ```
    This will launch the synthesis of audios with TokenSynth synthesizer then will assess it and will generate the results files and figures in the `./results/` folder.
    
    **Pipeline**:
    - Synthesis of the audio samples in `./data/synth/samples/`
    - Computing of the embeddings, saved in `./data/embeddings/`
    - Predictions and metrics by the model saved in `./results/synth/metrics/`

## Reproducing Results
To reproduce all results from the paper:
```bash
# Preprocess data
python ./data/preprocess_RWC.py --your_RWC_dataset_path

# Train timbre models
python ./experiments/timbre-model/main.py

# Assess synthesizer
python ./experiments/synthesizer-assessment/train_model.py
python ./experiments/synthesizer-assessment/main.py
```

## Results
Results are saved in the `./results/` directory:
- `timbre-model/metrics/`: Cross-validation predictions and metrics.
- `timbre-model/figures/`: Generated figures.
- `synth/metrics/`: Synthesizer assessment predictions and metrics.

## Citation
If you use this code, please cite our paper:
```bibtex
@article{chaslecauchy2026timbre,
  title={Timbre Trait Prediction for Performance Analysis of Musical Sound Synthesizer using Deep Neural Embeddings},
  author={Chasle Cauchy, Théo et al.},
  journal={Journal of Sound and Music Computing},
  year={2026}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open a GitHub issue or contact theo.chasle-cauchy@ls2n.fr.