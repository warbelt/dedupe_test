# Realiza todas las funciones de preprocesado sobre un campo
def preprocess(column):
    column = none_empty_fields(column)
    return column


# Sustituye campos vacíos del csv por None
def none_empty_fields(column):
    if not column:
        column = None
    return column
