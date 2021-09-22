def open_csv_at_given_line(path, first_line):
    import csv
    reader = csv.reader(open(path, mode='r'),
                        delimiter=';')
    for i in range(first_line):
        reader.__next__()
    header = reader.__next__()
    return header, reader

path = "C:/Users/sfenton/Code/Repositories/CO2footprint/DATA/settings.xlsx"

open_csv_at_given_line(path, 5)


"""def read_data_file():
    path =  "C:/Users/sfenton/Code/Repositories/CO2footprint/SCRIPTS/README"
    with open(path) as f:
        readl = f.readlines()
        d = dict()
        for line in readl:
            line.strip()
            line.split('=')
            print(line)
            key = str(line[0])
            val = line[1]
            d[key] = val
    print("1", readl)
    print(d)

a = read_data_file()

print(a)
#def export_data():
"""