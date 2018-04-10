from datetime import date


# Realiza todas las funciones de preprocesado sobre un campo
def preprocess(column):
    column = none_empty_fields(column)
    return column


# Sustituye campos vacíos del csv por None
def none_empty_fields(column):
    if not column:
        column = None
    return column


# Calcula la ruta total a la carpeta de deduplicación en función de la ruta base y la fecha
def calculate_path(base_path):
    # Fecha en formato DDMMYYYY
    today = date.today()
    today_string = today.strftime("%d%m%Y")

    # Si la ruta base no termina en "\", la añade al final antes de concatenar la fecha
    if base_path[-1] != "\\":
        base_path += "\\"

    return base_path + today_string
