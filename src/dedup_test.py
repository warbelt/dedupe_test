import csv

import cProfile
import dedupe

import src.dedup_config as config
from src.dedup_utils import *


def read_messy_data(filename):
    """
    Read in our data from a CSV file and create a dictionary of records,
    where the key is a unique record ID and each value is dict
    """

    try:
        f = open(filename, encoding=config.ENCODING)
    except IOError:
        print(filename + " no encontrado")
        raise IOError

    reader = csv.DictReader(f, delimiter=config.DELIMITER)
    data_d = {}
    for row in reader:
        clean_row = [(k, preprocess(v)) for (k, v) in row.items()]
        row_id = int(row['CONTACTO'])
        data_d[row_id] = dict(clean_row)

    return data_d


def write_clusters(clustered_dupes):
    # ## Writing Results

    # Write our original data back out to a CSV with a new column called
    # 'Cluster ID' which indicates which records refer to each other.

    cluster_membership = {}
    cluster_id = 0
    for (cluster_id, cluster) in enumerate(clustered_dupes):
        id_set, scores = cluster
        for record_id, score in zip(id_set, scores):
            cluster_membership[record_id] = {
                "cluster id": cluster_id,
                "confidence": score
            }

    singleton_id = cluster_id + 1

    with open(config.OUTPUT_FILE, 'w', encoding=config.ENCODING) as f_output, \
            open(config.INPUT_FILE, encoding=config.ENCODING) as f_input:

        writer = csv.writer(f_output)
        reader = csv.reader(f_input, delimiter=config.DELIMITER)

        heading_row = next(reader)
        heading_row.insert(0, 'confidence_score')
        heading_row.insert(0, 'Cluster ID')

        writer.writerow(heading_row)

        for row in reader:
            row_id = int(row[0])
            if row_id in cluster_membership:
                cluster_id = cluster_membership[row_id]["cluster id"]
                row.insert(0, cluster_membership[row_id]['confidence'])
                row.insert(0, cluster_id)

            else:
                row.insert(0, None)
                row.insert(0, singleton_id)
                singleton_id += 1

            writer.writerow(row)


def active_training():
    print("MODE: Active training")

    try:
        data_d = read_messy_data(config.INPUT_FILE)
    except IOError:
        print("No se pudo abrir el fichero de records de entrada - " + config.INPUT_FILE)
        raise

    deduper = dedupe.Dedupe(config.FIELDS)
    deduper.sample(data_d, config.SAMPLE_SIZE)
    dedupe.consoleLabel(deduper)

    with open(config.TRAINING_FILE, 'w') as tf:
        deduper.writeTraining(tf)


def deduplicate():
    print("MODE: Deduplicate")

    try:
        data_d = read_messy_data(config.INPUT_FILE)
    except IOError:
        print("No se pudo abrir el fichero de records de entrada - " + config.INPUT_FILE)
        raise

    deduper = dedupe.Dedupe(config.FIELDS)

    deduper.sample(data_d, config.SAMPLE_SIZE)
    dedupe.consoleLabel(deduper)

    deduper.train(config.USE_INDEX_PREDICATES)

    with open(config.TRAINING_FILE, 'w') as tf:
        deduper.writeTraining(tf)
    with open(config.SETTINGS_FILE, 'wb') as sf:
        deduper.writeSettings(sf)

    threshold = deduper.threshold(data_d, recall_weight=config.RECALL_WEIGHT)
    clustered_dupes = deduper.match(data_d, threshold)

    write_clusters(clustered_dupes)


def main():
    try:
        {
            "test": deduplicate,
            "active_training": active_training,
        }[config.DEDUPE_MODE]()
    except IOError:
        print("Ejecución cancelada")


if __name__ == "__main__":
    if config.PROFILING:
        pr = cProfile.Profile()
        pr.enable()

        main()

        pr.disable()
        pr.dump_stats(config.PROFILING_FILE)
    else:
        main()
