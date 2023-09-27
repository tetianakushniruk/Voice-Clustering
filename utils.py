from collections import Counter

import librosa
import numpy as np
import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def extract_features(signal, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    return mfccs, zcr


def flatten_features(features):
    return np.concatenate([f.flatten() for f in features])


def get_audio_data(audio_folder, segment_length=2):
    data = []
    for file_name in os.listdir(audio_folder):
        audio_path = os.path.join(audio_folder, file_name)
        audio, sr = librosa.load(audio_path)

        num_segments = len(audio) // (segment_length * sr)

        for i in range(num_segments):
            start_time = i * segment_length
            end_time = (i + 1) * segment_length
            segment = audio[start_time * sr:end_time * sr]

            extracted_features = extract_features(segment, sr)

            features = flatten_features(extracted_features)
            segment_info = {
                "features": features,
                "file_name": file_name,
                "segment_number": i
            }

            data.append(segment_info)
    features = np.array([x['features'] for x in data])
    return data, features


def clustering(X, n_clusters=None, distance_threshold=None, n_pca_components=0.99):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = min_max_scaler.fit_transform(X)

    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X_scaled)

    model = AgglomerativeClustering(n_clusters=n_clusters,
                                    distance_threshold=distance_threshold)
    return model.fit_predict(X_pca)


def labels_to_dict(labels, data):
    cluster_mapping = {}
    most_common_clusters = {}

    for i, segment_info in enumerate(data):
        file_name = segment_info['file_name']
        segment_number = segment_info['segment_number']
        cluster_label = labels[i]

        if file_name not in cluster_mapping:
            cluster_mapping[file_name] = {}

        cluster_mapping[file_name][segment_number] = cluster_label

    for file_name, segment_clusters in cluster_mapping.items():
        cluster_counts = Counter(segment_clusters.values())

        most_common_cluster = cluster_counts.most_common(1)[0][0]

        most_common_clusters[most_common_cluster] = most_common_clusters.get(
            most_common_cluster, []
        )
        most_common_clusters[most_common_cluster].append(file_name)

    return {str(cluster): file_list for cluster, file_list in most_common_clusters.items()}