# Timbre Trait Prediction for Performance Analysis of Musical Sound Synthesizer using Deep Neural Embeddings

**Description**: Code repository for the paper "Timbre Trait Prediction for Performance Analysis of Musical Sound Synthesizer using Deep Neural Embeddings". This repository contains scripts to reproduce the experiments presented in the paper, the training of the timbre model and the assessment of the synthesizer.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/TheoChasleCauchy/paperTTPSynthesizerAssessment.git
   cd paperTTPSynthesizerAssessment
   ```

2. Create a virtual environment (optional)
   ```bash
   # If you use uv (Recommended)
   uv sync

   # If you don't use uv
   # Make sure you have an installed Python version between 3.11 and 3.12 
   python -m venv venv
   pip install . 

   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install requirements:
    ```bash
    uv sync # If you use uv (Recommended)
    pip install . 
    ```

## Dataset
Download the [RWC Dataset](https://zenodo.org/records/17170844) and place it in `data/RWC/`.

## Configuration
- Experiment parameters (e.g., train-valid split, models architectures, embeddings used, training parameters) are defined in:
  - `data/RWC/metadata/split_config.yaml`
  - `experiments/timbre-model/config.yaml`
  - `experiments/synthesizer_assessment/config.yaml`
- Edit these files to customize the experiments.

## Experiments

1. Process the RWC samples:
    ```bash
    python process_RWC/process_RWC.py
    ```
    This will launch the process pipeline of RWC, split the RWC audio files into single notes samples available in `data/RWC/RWC_processed/`, then compute the embeddings (CLAP, CLAP Music, VGGish and MERT) and save them in `data/embeddings/`.
    
    **Pipeline**:
    - Split of the RWC audio files into single notes samples available in `data/RWC/RWC_processed/`
    - Creation of a metadata CSV file of the RWC samples saved in `data/RWC/metadata/`
    - Computation of the embeddings (CLAP, CLAP Music, VGGish and MERT) and save them in `data/RWC/embeddings/`
    - Creation of a metadata CSV file of the RWC embeddings saved in `data/RWC/metadata/`
    - Creation of a train-valid split (Two distinct CSV files) depending on the split in split_config.yaml.
    
        NB: If you want to change the split, use :
        ```bash
        python process_RWC/change_split.py # If you want to specify yourself the split in split_change.yaml
        python process_RWC/change_split.py -r # If you want to randomly change thesplit in split_change.yaml
        python process_RWC/change_split.py -r --train_proportion # If you want specify the train split proportion (default 0.8)
        python process_RWC/change_split.py -r --random_seed # If you want specify the random seed (default 1) 
        ```

2. Launch the timbre-model pipeline to train the models
    ```bash
    python experiments/cross-validation_timbre_model/scripts/main.py
    ```
    This will launch the training of all the models considering the parameters in `experiments/cross-validation_timbre_model/config.yaml` and will generate the results files and figures in the `experiments/cross-validation_timbre_model/results` folder.
    
    **Pipeline**:
    - Train each model with a cross-validation approach, saved in `models/cross-validation_timbre_model/`
    - Compute the cross-validation predictions and metrics, saved in `experiments/cross-validation_timbre_model/results/`
    - Generation of figures, saved in `experiments/cross-validation_timbre_model/results/figures/`


3. Launch the synthesizer_assessment pipeline assess the synthesizer

    ```bash
    python experiments/synthesizer_assessment/main.py
    ```
    This first train a model on all instruments by specifing the parameters in the `experiments/synthesizer_assessment/config.yaml` file (by default the model is trained on CLAP emneddings with no hidden layers). Then it will launch the synthesis of audios with TokenSynth synthesizer then will assess it and will generate the results files and figures in the `results/` folder.
    
    **Pipeline**:
    - Training of the model on all instruments saved in `models/synthesizer_assessment/`.
    - Computing of the mean embedding of RWC samples for each instrument `RWC/mean_embeddings/clap_embeddings/`.
    - Generation of midi files for each note in `data/TokenSynth/midi_files`
    - Synthesis of the audio samples in `data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/`.
    - Computing of the embeddings, saved in `data/TokenSynth/Embeddings/{condition_type}_conditioned/clap_embeddings/`.
    - Predictions, metrics and generated figures are saved in `experiments/synthesizer_assessment/results/`.

## Reproducing Results
To reproduce all results from the paper:
```bash
# Preprocess data
python data/preprocess_RWC.py --your_RWC_dataset_path

# Train timbre models
python experiments/cross-validation_timbre_model/main.py

# Assess synthesizer
python experiments/synthesizer_assessment/main.py
```

## Results
Results are saved in the `results/` sub-directories of the experiments directories:
- `experiments/cross-validation_timbre_model/results/`: Cross-validation predictions, metrics and generated figures.
- `experiments/synthesizer_assessment/results/`: Predictions and metrics on synthesized samples.

## Companion page
Please check the companion page for audio examples: https://theochaslecauchy.github.io/paperTTPSynthesizerAssessment/

## Citation
If you use this code, please cite our paper:
```bibtex
@article{chaslecauchy2026timbre,
  title={Timbre Trait Prediction for Performance Analysis of Musical Sound Synthesizer using Deep Neural Embeddings},
  author={Chasle Cauchy, Théo et al.},
  conference={Conference of Sound and Music Computing},
  year={2026}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open a GitHub issue or contact theo.chasle-cauchy@ls2n.fr.