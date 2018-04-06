# ######################
# ###     GENERAL    ###
# ######################

# Nombre del test. Determina el nombre de los ficheros de salida para diferenciarlos de otras ejecuciones
EXECUTION_NAME = "train_ip_ext_001"

# Modo de ejecución:
#   - "test" : completa con sample, active training, train, match
#   - "active_training" : sample, active training. Guarda labels en fichero
DEDUPE_MODE = "active_training"

# Tipo de codificación de documentos de entrada
ENCODING = "latin_1"

# Cadena delimitadora de csv
DELIMITER = ';'

# Activar el profiling con cProfile
PROFILING = True


# ######################
# ###     RUTAS      ###
# ######################

# Carpeta para almacenar ficheros profiling
PROFILING_FILE = "../profiling/" + EXECUTION_NAME + ".prof"

# Contiene los records a deduplicar
INPUT_FILE = "../dataset/CRM_CONT.CSV"

# Fichero de resultados
OUTPUT_FILE = "../output/clusters_" + EXECUTION_NAME + ".csv"

# Contiene el modelo entrenado y los predicados
SETTINGS_FILE = "../output/settings_" + EXECUTION_NAME

# Contiene los records etiquetados a mano con el entrenamiento activo
TRAINING_FILE = "../output/training_" + EXECUTION_NAME


# ######################
# ###     DEDUPE     ###
# ######################

# Definición de variables de Dedupe
# Lista de diccionarios, cada diccionario se corresponde con una variable
# Para tipos de variables y parámetros opcionales:
# http://dedupe.readthedocs.io/en/latest/Variable-definition.html
FIELDS = [
    {"field": "NOMB", "type": "String"},
    {"field": "APE1", "type": "String"},
    {"field": "APE2", "type": "String"},
    {"field": "MAIL", "type": "String"},
    {"field": "TFFI", "type": "String"},
]

# Tamaño de muestra para el muestreo
SAMPLE_SIZE = 15000

# Usar index predicates en el training o no
# Index predicates son más caros en memoria y computacionalmente. Además no se pueden paralelizar porque se rompe
# el índice. Mejoran la eficacia del blocking para hacer menos comparaciones
INDEX_PREDICATES = True

# Peso del recall frente a la precisión para calcular el umbral de la regresión logística
# score = recall * precision / (recall + recall_weight ** 2 * precision)
# https://github.com/dedupeio/dedupe/blob/master/dedupe/api.py [86]
RECALL_WEIGHT = 1
