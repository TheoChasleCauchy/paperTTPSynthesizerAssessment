import argparse

def main():
    parser = argparse.ArgumentParser(description="Train timbre models with configuration in config.yaml and compute metrics.")

    args = parser.parse_args()

if __name__ == "__main__":
    main()