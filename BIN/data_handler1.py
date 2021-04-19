

STR_FEATURES = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding', 'BREEAM Rating']
INT_FEATURES = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN/m2)']
FEATURES_NAMES = STR_FEATURES + INT_FEATURES
OUTPUT_NAMES = ['Calculated Total tCO2e', 'Calculated tCO2e/m2']


def open_csv_at_given_line(filename, first_line, delimiter):
    import csv
    reader = csv.reader(open(filename + '.csv', mode='r'), delimiter=delimiter)
    for i in range(first_line):
        reader.__next__()
    header = reader.__next__()
    return header, reader


def index_dict_from_csv(filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    CST = dict()
    for f in FEATURES_NAMES:
        CST[f] = []
    for line in reader:
        for f in FEATURES_NAMES:
            index = header.index(f)
            if line[index] not in CST[f]:
                CST[f].append(line[index])
    return CST


def split_X_Y_values(X_NAMES, Y_NAMES, filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    X_values, Y_values = [], []
    for line in reader:
        for (names, values) in [(X_NAMES, X_values), (Y_NAMES, Y_values)]:
            values.append([line[header.index(name)] for name in names])
    return X_values, Y_values
    # line[header.index(name)] = value in that column
"""
def split_X_Y_values(X_NAMES, Y_NAMES, filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)

    X_values = []
    for x_name in X_NAMES:
        print(x_name)
        fx_values = []
        for line in reader:
            fx_values.append(line[header.index(x_name)])
            print(fx_values)
        X_values.append(fx_values)

    Y_values = []
    for y_name in Y_NAMES:
        fy_values = []
        for line in reader:
            fy_values.append(line[header.index(y_name)])
        Y_values.append(fy_values)

    return X_values, Y_values
"""

"""def split_X_Y_values(X_NAMES, Y_NAMES, filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)

    X_values = []
    for f in X_NAMES:
        fx_values = []
        for line_value in reader:
            fx_values.append(line_value)
        X_values.append(fx_values)

    Y_values = []
    for f in Y_NAMES:
        fy_values = []
        for line_value in reader:
            fy_values.append(line_value)
        Y_values.append(fy_values)


    return X_values, Y_values
"""
xd, yd = split_X_Y_values(FEATURES_NAMES, OUTPUT_NAMES, "DATA/210413_PM_CO2_data", 5, delimiter=';')

print ("xd", len(xd), type(xd), xd)
for i in range(len(xd)):
    print (i, len(xd[i]), type(xd[i]))

print ("yd", len(yd), type(yd))
for i in range(len(yd)):
    print (i, len(yd[i]), type(yd[i]))

for elem in xd[11]:
    print(elem)
index_dicty = index_dict_from_csv("DATA/210413_PM_CO2_data", 5)
print(index_dicty.keys())

def translate_inputs(header, line):
    """
    Comments :
    - line[header.index(feature_name)] = used to query a feature_value

    --------------------------------------------------------------------------------------------------------------------

    input : feature name and training example line
    output : translated line

    """

    x = []
    for feature_name in STR_FEATURES:
        x = x + translate_str_feature(feature_name, line[header.index(feature_name)])
    for feature_name in INT_FEATURES:
        x = x + translate_int_feature(feature_name, line[header.index(feature_name)])
    return x

def translate_outputs(header, line):
    """

    input : feature name and training example line
    output : translated output line

    """

    y = []
    for output_name in INT_OUTPUTS:
        y = y + translate_int_feature(output_name, line[header.index(output_name)])
        # line[header.index(output_name) = ex 3200kgCO2e
    return y


def translate_str_feature(dictionary, feature_name, feature_value):

    return dictionary[feature_name].index(feature_value)

"""
test = translate_str_feature(dicty, 'Sector', dicty['Sector'][0])
#print(test)

for value in dicty['Sector']:
    print(translate_str_feature(dicty, 'Sector', value))
"""

def extract_data_from_csv(filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    X, Y = [], []
    for line in reader:
        X.append(translate_inputs(header, line))
        Y.append(translate_outputs(header, line))
    return X, Y

def translate_data():
    return translate_inputs(X), translate_inputs(y)

def format_data_to_df(data_matrix):
    dim = len(data_matrix[0])
    num_samples, num_features = len(data_matrix), len(data_matrix[0])
    data = np.zeros((num_samples, num_features))
    for i in range(num_samples):
        data[i, :] = np.array([float(x) for x in data_matrix[i].strip().split(";")])
    return data