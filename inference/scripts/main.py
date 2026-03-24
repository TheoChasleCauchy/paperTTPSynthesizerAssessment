import argparse
from compute_embeddings import compute_embeddings
from inference_on_audio_embeddings import inference

def main(audios_folder: str, model_save_folder: str):
    # Compute embeddings for the audio files
    compute_embeddings(audios_folder)

    # Perform inference on the computed embeddings
    inference(audios_folder, model_save_folder)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compute features for audio files.")
    parser.add_argument("audios_folder", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--model_save_folder", type=str, default=None, help="Path to the folder containing the saved model (optional)")

    # Parse arguments
    args = parser.parse_args()

    if not args.model_save_folder:
        model_save_folder = "models/paper_checkpoint"
    else:
        model_save_folder = args.model_save_folder

    # Call main with the directory argument
    main(args.audios_folder, model_save_folder)