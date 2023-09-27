import argparse
import json
import os
from utils import get_audio_data, clustering, labels_to_dict


def main():
    parser = argparse.ArgumentParser(description="Audio Clustering")
    parser.add_argument("--folder_path",
                        required=True,
                        help="Path to the folder containing audio files.")
    parser.add_argument("--output_path",
                        required=False,
                        help="Path to the output file.")

    args = parser.parse_args()

    folder_path = args.folder_path

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder '{folder_path}' not found.")

    data, X = get_audio_data(folder_path)
    labels = clustering(X, n_clusters=3)
    results = labels_to_dict(labels, data)

    output_path = args.output_path
    if not output_path:
        output_path = 'results.json'

    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f'Results are saved in {output_path}\n')
    print(results)


if __name__ == "__main__":
    main()
