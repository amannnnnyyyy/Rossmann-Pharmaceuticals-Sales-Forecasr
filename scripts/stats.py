def missing_values(data):
    result = {}
    for key in data:
        error_count= data[key].isna().sum()
        result[key] = error_count
    return result