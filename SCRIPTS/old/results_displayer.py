import csv

def from_dict_to_list_of_features_powers(dico, virtual_label_list):
    out = ['ones']
    for feature in virtual_label_list:
        for power in dico[feature]:
            out.append(feature + " ^" + str(power))
    return out


def dictionary_of_results(focus, reg, labels, theta_opt, mse, mae, cost):
    results = dict()
    results["object of study"] = [focus]
    results ["regularization param"] = [reg]
    results ["polynomial_exponents"] = [labels[index]for index in range(len(theta_opt))]
    results ["theta_opt"] = [theta_opt[index]for index in range(len(theta_opt))]
    results["Mean square error"] = [mse]
    results["Mean absolute error"] = [mae]
    results["Cost"] = [cost]
    return results


def print_results(res_dict):
    for k, v in res_dict.items():
        print(k,v)


def export_results_as_csv(path, res_dict, filename="results"):

    from itertools import zip_longest
    d = [res_dict[key] for key in res_dict.keys()]
    export_data = zip_longest(*d, fillvalue='')
    with open(path + filename + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(res_dict.keys())
        wr.writerows(export_data)
    myfile.close()


def csv_into_columns(input_csv, path, filename="results.txt"):  # TODO : doesn't work..
    with open(input_csv + '.csv') as inf:
        with open(path + filename + '.csv', 'w') as outf:
            for line in inf:
                outf.write('\\'.join(line.split(',')))


def export_results_as_text(filename, res_dict):

    fo = open(filename + "results.txt", 'a')
    for k, v in res_dict.items():
        if len(res_dict[k]) <= 1:
            fo.write(str(k) + ' >>> '+ str(v) + '\n\n')
        if len(res_dict[k]) > 1:
            fo.write(str(k) + ' >>> ' + '\n\n')
            for i in range(len(res_dict[k])):
                fo.write(str(i) + '  '+ str(res_dict[k][i]) + '\n\n')
    fo.close()


def export_results_as_text_adv(path, res_dict, key_a, key_b, filename="results"):

    fo = open(path + filename + ".txt", 'w')
    for k, v in res_dict.items():
        if len(res_dict[k]) <= 1:
            fo.write(str(k) + ' : '+ str(res_dict[k][0]) + '\n\n')

    a = [key_a]+res_dict[key_a]
    b = [key_b]+res_dict[key_b]
    c = [a, b]
    for x in zip(*c):
        fo.write("{0}\t{1}\n".format(*x))
    fo.close()


def export_results_as_json(path, res_dict, filename="results"):
    import json
    with open(path + filename +"_json_fmt" + ".csv", 'w') as file:
        file.write(json.dumps(res_dict))
    file.close()
