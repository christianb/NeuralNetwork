def read_file(file_name):
    file = open(file_name, 'r')
    data = file.readlines()  # do not read whole files in memory!
    file.close()

    return data
    pass
