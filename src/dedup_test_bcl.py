import csv
import dedupe
import time
import cProfile
from collections import OrderedDict

input_file = "../dataset/20180228_EXTRACCION_CLIENTES_BARCELO.CSV"
output_file = "../output/deduped_cont_bcl.csv"
settings_file = "../output/settings_bcl"
training_file = "../output/training_bcl"

fields = [
    {"field": "MEMBER_ID", "type": "String"},
    {"field": "FECHA_ALTA", "type": "String"},
    {"field": "NOMBRE", "type": "String", "has_missing": True},
    {"field": "APELLIDOS", "type": "String", "has_missing": True},
    {"field": "EMAIL", "type": "String"},
]


class Timer:
    def __init__(self):
        self.start_time = time.clock()
        self._recorded_times = OrderedDict()

    def step(self, name="Unknown"):
        elapsed = time.clock() - self.start_time
        self.start_time = time.clock()
        self._recorded_times[name] = elapsed

    def print_times(self):
        print(self._recorded_times)


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
    with open(filename, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")

        lines = 0
        max_lines = 150000

        for row in reader:
            clean_row = [(k, preprocess(v)) for (k, v) in row.items()]
            row_id = int(row['MEMBER_ID'])
            data_d[row_id] = dict(clean_row)

            lines+=1
            if lines > max_lines: break

    return data_d


# COPIADO DE examples.py
# CUIDADO!
def write_clusters(clustered_dupes, data_d):
    # ## Writing Results

    # Write our original data back out to a CSV with a new column called
    # 'Cluster ID' which indicates which records refer to each other.

    cluster_membership = {}
    cluster_id = 0
    for (cluster_id, cluster) in enumerate(clustered_dupes):
        id_set, scores = cluster
        cluster_d = [data_d[c] for c in id_set]
        canonical_rep = dedupe.canonicalize(cluster_d)
        for record_id, score in zip(id_set, scores):
            cluster_membership[record_id] = {
                "cluster id": cluster_id,
                "canonical representation": canonical_rep,
                "confidence": score
            }

    singleton_id = cluster_id + 1

    with open(output_file, 'w') as f_output, open(input_file) as f_input:
        writer = csv.writer(f_output)
        reader = csv.reader(f_input)

        heading_row = next(reader)
        heading_row.insert(0, 'confidence_score')
        heading_row.insert(0, 'Cluster ID')
        canonical_keys = canonical_rep.keys()
        for key in canonical_keys:
            heading_row.append('canonical_' + key)

        writer.writerow(heading_row)

        for row in reader:
            row_id = int(row[0])
            if row_id in cluster_membership:
                cluster_id = cluster_membership[row_id]["cluster id"]
                canonical_rep = cluster_membership[row_id]["canonical representation"]
                row.insert(0, cluster_membership[row_id]['confidence'])
                row.insert(0, cluster_id)
                for key in canonical_keys:
                    row.append(canonical_rep[key].encode('utf8'))
            else:
                row.insert(0, None)
                row.insert(0, singleton_id)
                singleton_id += 1
                for key in canonical_keys:
                    row.append(None)
            writer.writerow(row)


def main():
    timer = Timer()
    data_d = read_data(input_file)
    timer.step("Load data")

    deduper = dedupe.Dedupe(fields)
    timer.step("Initialize dedupe")

    # cProfile.runctx('deduper.sample(data_d, 1000)', globals(), locals(), filename = "../profiling/Dedupe.sample")
    deduper.sample(data_d, 15000)
    timer.step("Sample")

    dedupe.consoleLabel(deduper)
    timer.step("Label")

    deduper.train(index_predicates=False)
    timer.step("Train")

    with open(training_file, 'w') as tf:
        deduper.writeTraining(tf)
    with open(settings_file, 'wb') as sf:
        deduper.writeSettings(sf)
    timer.step("Save settings")

    threshold = deduper.threshold(data_d, recall_weight=1)
    timer.step("Calculate threshold")

    clustered_dupes = deduper.match(data_d, threshold)
    timer.step("Calculate clusters")

    # write_clusters(clustered_dupes, data_d)
    # timer.step("Write clustered data")

    timer.print_times()


if __name__ == "__main__":
    main()
