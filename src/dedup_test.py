import csv
from datetime import date

import cProfile
import dedupe

import src.dedup_config as CONFIG
from src.dedup_utils import *


# Calcula la ruta total a la carpeta de deduplicación en función de la ruta base y la fecha
def calculate_path():
    # Fecha en formato DDMMYYYY
    today = date.today()
    today_string = today.strftime("%d%m%Y")

    # Si la ruta base no termina en "\", la añade al final antes de concatenar la fecha
    base_path = CONFIG.GENERAL.BASE_PATH
    if base_path[-1] != "\\":
        base_path += "\\"

    return base_path + today_string


# Lee los datos a deduplicar de un CSV y crea un diccionario de records.
# La clave es un record ID único y el valor es un diccionario con los datos
# Preprocesa los campos
def read_messy_data(filename):
    """
    Read in our data from a CSV file and create a dictionary of records,
    where the key is a unique record ID and each value is dict
    """

    try:
        f = open(filename, encoding=CONFIG.GENERAL.ENCODING)
    except IOError:
        print(filename + " no encontrado")
        raise IOError

    reader = csv.DictReader(f, delimiter=CONFIG.GENERAL.DELIMITER)
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

    with open(CONFIG.PATHS.OUTPUT_FILE, 'w', encoding=CONFIG.GENERAL.ENCODING) as f_output, \
            open(CONFIG.PATHS.INPUT_FILE, encoding=CONFIG.GENERAL.ENCODING) as f_input:

        writer = csv.writer(f_output)
        reader = csv.reader(f_input, delimiter=CONFIG.GENERAL.DELIMITER)

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


# Función para realizar entrenamiento activo (clasificación manual de pares)
# Después de finalizar guarda todos los casos en un fichero y termina el proceso
def active_training():
    print("MODE: Active training")

    # carga fichero de datos a deduplicar
    try:
        data_d = read_messy_data(CONFIG.PATHS.INPUT_FILE)
    except IOError:
        print("No se pudo abrir el fichero de records de entrada - " + CONFIG.PATHS.INPUT_FILE)
        raise

    # Entrenamiento Activo
    deduper = dedupe.Dedupe(CONFIG.DEDUPE.FIELDS)
    deduper.sample(data_d, CONFIG.DEDUPE.SAMPLE_SIZE)
    dedupe.consoleLabel(deduper)

    with open(CONFIG.PATHS.TRAINING_FILE, 'w') as tf:
        deduper.writeTraining(tf)


# Función principal de deduplicación. Realiza todas las tareas
def deduplicate():
    print("MODE: Deduplicate")

    # Carga fichero de datos a deduplicar
    try:
        data_d = read_messy_data(CONFIG.PATHS.INPUT_FILE)
    except IOError:
        print("No se pudo abrir el fichero de records de entrada - " + CONFIG.PATHS.INPUT_FILE)
        raise

    # Inicializa objeto dedupe
    deduper = dedupe.Dedupe(CONFIG.DEDUPE.FIELDS)

    # Muestreo y entrenamiento activo
    deduper.sample(data_d, CONFIG.DEDUPE.SAMPLE_SIZE)
    dedupe.consoleLabel(deduper)

    # Entrenamiento de modelo predictivo (por defecto regresión logística
    deduper.train(CONFIG.DEDUPE.USE_INDEX_PREDICATES)

    # Guarda
    with open(CONFIG.PATHS.TRAINING_FILE, 'w') as tf:
        deduper.writeTraining(tf)
    with open(CONFIG.PATHS.SETTINGS_FILE, 'wb') as sf:
        deduper.writeSettings(sf)

    # Calcula umbral para la regresión logistica
    threshold = deduper.threshold(data_d, recall_weight=CONFIG.DEDUPE.RECALL_WEIGHT)
    # Agrupación de matches en clusters
    clustered_dupes = deduper.match(data_d, threshold)

    write_clusters(clustered_dupes)


def main():
    try:
        {
            "test": deduplicate,
            "active_training": active_training,
        }[CONFIG.GENERAL.DEDUPE_MODE]()
    except IOError:
        print("Ejecución cancelada")


if __name__ == "__main__":
    if CONFIG.GENERAL.PROFILING:
        pr = cProfile.Profile()
        pr.enable()

        main()

        pr.disable()
        pr.dump_stats(CONFIG.PATHS.PROFILING_FILE)
    else:
        main()
