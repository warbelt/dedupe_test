import csv
from collections import OrderedDict

import cProfile
import dedupe
import dedup_settings as settings


def preprocess(column):
    if not column:
        column = None
    return column


def read_data(filename):
    """
    Read in our data from a CSV file and create a dictionary of records,
    where the key is a unique record ID and each value is dict
    """

    data_d = {}
    with open(filename, encoding=settings.ENCODING) as f:
        reader = csv.DictReader(f, delimiter=settings.DELIMITER)
        for row in reader:
            clean_row = [(k, preprocess(v)) for (k, v) in row.items()]
            row_id = int(row['CONTACTO'])
            data_d[row_id] = dict(clean_row)

    return data_d


def write_clusters(clustered_dupes, data_d):
    # ## Writing Results

    # Write our original data back out to a CSV with a new column called
    # 'Cluster ID' which indicates which records refer to each other.

    cluster_membership = {}
    cluster_id = 0
    for (cluster_id, cluster) in enumerate(clustered_dupes):
        id_set, scores = cluster
        cluster_d = [data_d[c] for c in id_set]
        for record_id, score in zip(id_set, scores):
            cluster_membership[record_id] = {
                "cluster id": cluster_id,
                "confidence": score
            }

    singleton_id = cluster_id + 1

    with open(settings.OUTPUT_FILE, 'w', encoding=settings.ENCODING) as f_output, \
            open(settings.INPUT_FILE, encoding=settings.ENCODING) as f_input:

        writer = csv.writer(f_output)
        reader = csv.reader(f_input, delimiter=settings.DELIMITER)

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

    data_d = read_data(settings.INPUT_FILE)
    deduper = dedupe.Dedupe(settings.FIELDS)
    deduper.sample(data_d, settings.SAMPLE_SIZE)
    dedupe.consoleLabel(deduper)

    with open(settings.TRAINING_FILE, 'w') as tf:
        deduper.writeTraining(tf)


def deduplicate():
    print("MODE: Deduplicate")

    data_d = read_data(settings.INPUT_FILE)

    deduper = dedupe.Dedupe(settings.FIELDS)

    deduper.sample(data_d, settings.SAMPLE_SIZE)

    dedupe.consoleLabel(deduper)

    deduper.train(settings.INDEX_PREDICATES)

    with open(settings.TRAINING_FILE, 'w') as tf:
        deduper.writeTraining(tf)
    with open(settings.SETTINGS_FILE, 'wb') as sf:
        deduper.writeSettings(sf)

    threshold = deduper.threshold(data_d, recall_weight=settings.RECALL_WEIGHT)

    clustered_dupes = deduper.match(data_d, threshold)

    write_clusters(clustered_dupes, data_d)


def main():
    result = {
        "test": deduplicate,
        "active_training": active_training,
    }[settings.DEDUPE_MODE]()

    return result


if __name__ == "__main__":
    if settings.PROFILING:
        pr = cProfile.Profile()
        pr.enable()

        main()

        pr.disable()
        pr.dump_stats(settings.PROFILING_FILE)
    else:
        main()
