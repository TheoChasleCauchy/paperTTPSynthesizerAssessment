from preprocess_RWC import preprocess_RWC
from create_RWC_metadata import create_RWC_metadata
from create_embeddings_metadata import create_embeddings_metadata
from compute_split_metadata import split_metadata
from samples_to_embeddings import compute_embeddings


def main():
    preprocess_RWC()
    create_RWC_metadata()
    compute_embeddings()
    create_embeddings_metadata()
    split_metadata()


if __name__ == "__main__":
    main()