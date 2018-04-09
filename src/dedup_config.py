class GENERAL:
    # Ruta base para la deduplicación dentro del datalake.
    # A partir de esta ruta se calculan todos los nombres de dicheros si CALCULATE_PATHS == True
    # TODO: Path base final
    BASE_PATH = "adl:\\we\\dedup"

    # Nombre del test. Determina el nombre de los ficheros de salida para diferenciarlos de otras ejecuciones
    # Solo aplica cuando las rutas no son calculadas
    EXECUTION_NAME = "train_ip_ext_001"

    # TODO: Buscar solución más elegante para separar pruebas y produccion
    # Modo de ejecución:
    #   - "test" : completa con sample, active training, train, match
    #   - "active_training" : sample, active training. Guarda labels en fichero
    DEDUPE_MODE = "test"

    # Tipo de codificación de documentos de entrada
    ENCODING = "latin_1"

    # Cadena delimitadora de csv
    DELIMITER = ';'

    # Generar profiling con cProfile.
    # El profiling generado se guarda en PROFILING_FILE
    # Permite visualizar una pila de llamadas y tiempos. Recomendado usar Snakeviz para analizar el profiling
    # Para usar Snakeviz hay que activar antes el virtual environment de la deduplicación
    PROFILING = False

    # Cargar los casos clasificados en entrenamientos activos anteriores
    # Se carga desde el fichero TRAINING_FILE
    # Se puede cargar y realizar una nueva fase de entrenamiento activo sobre las anteriores para ampliarla
    LOAD_TRAINING = True

    # Cargar modelo entrenado y predicados
    # Se carga desde el fichero SETTINGS_FILE
    # Si se carga el modelo y los predicados no se debe repetir el proceso de train()
    LOAD_SETTINGS = False

    # Calcular las rutas en función de la fecha actual a partir de una ruta base
    # Si es False, usa las rutas escritas en la sección RUTAS
    CALCULATE_PATHS = False


class PATHS:
    # Carpeta para almacenar ficheros profiling
    PROFILING_FILE = "../profiling/" + GENERAL.EXECUTION_NAME + ".prof"

    # Contiene los records a deduplicar
    INPUT_FILE = "../dataset/CRM_CONT.CSV"

    # Fichero de resultados
    OUTPUT_FILE = "../output/clusters_" + GENERAL.EXECUTION_NAME + ".csv"

    # Contiene los records etiquetados a mano con el entrenamiento activo
    TRAINING_FILE = "../output/training_" + GENERAL.EXECUTION_NAME + ".json"

    # Contiene el modelo entrenado y los predicados
    SETTINGS_FILE = "../output/settings_" + GENERAL.EXECUTION_NAME


class DEDUPE:
    # Definición de variables de Dedupe
    # Lista de diccionarios, cada diccionario se corresponde con una variable
    # Para tipos de variables y parámetros opcionales:
    # http://dedupe.readthedocs.io/en/latest/Variable-definition.html
    # TODO: [] Acordar campos, tipos y hasEmpty
    FIELDS = [
        {"field": "NOMB", "type": "String"},
        {"field": "APE1", "type": "String"},
        {"field": "APE2", "type": "String"},
        {"field": "MAIL", "type": "String"},
        {"field": "TFFI", "type": "String"},
    ]

    # Cantidad de registros muestreados para el entrenamiento
    SAMPLE_SIZE = 15000

    # Usar index predicates en el training o no
    # Index predicates son más caros en memoria y computacionalmente. Además no se pueden paralelizar porque se rompe
    # el índice. Mejoran la eficacia del blocking para hacer menos comparaciones
    USE_INDEX_PREDICATES = True

    # Peso del recall frente a la precisión para calcular el umbral de la regresión logística
    # score = recall * precision / (recall + recall_weight ** 2 * precision)
    # https://github.com/dedupeio/dedupe/blob/master/dedupe/api.py [86]
    # TODO: [] Acordar peso
    RECALL_WEIGHT = 1
