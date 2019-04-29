import os.path


def read_file(file_name):
    file = open(file_name, 'r')
    data = file.readlines()  # do not read whole files in memory!
    file.close()

    return data
    pass


def read_optional_file_or_default(file_name_optional, file_name_default):
    optional_data_file = file_name_optional
    if os.path.isfile(optional_data_file):
        return read_file(optional_data_file)
    else:
        return read_file(file_name_default)
        pass

    pass
