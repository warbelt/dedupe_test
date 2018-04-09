def preprocess(column):
    column = none_empty_fields(column)
    return column


def none_empty_fields(column):
    if not column:
        column = None
    return column