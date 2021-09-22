import csv

class archive_implementation():

    def __init__(self, defined_model, preprocessed_data, processed_data, tuned_regression, displayed_data, displayed_results):

        #super().__init__() #TODO: ?
        self.model = defined_model
        self.preprocessed_data = preprocessed_data
        self.processed_data = processed_data
        self.tuned_regression = tuned_regression
        self.displayed_data = displayed_data
        self.displayed_results = displayed_results

    def create_output_folder(self, save = True):
        import os
        outputPath = self.model.output_path
        if save and not os.path.isdir(outputPath):
            os.makedirs(outputPath)

    def export_results_as_csv(self, filename="results", save=True):
        self.create_output_folder(save=save)
        res_dict = self.displayed_results.dictionary_of_results()
        path = self.model.output_path
        short_k = []
        short_v = []
        long_k = []
        long_v = []
        for k, v in res_dict.items():
            if len(v) <= 1:
                short_k.append(k)
                short_v.append(v)
            else:
                long_k.append(k)
                long_v.append(v)
        with open(path + filename + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(short_k)
            wr.writerow(short_v)
            wr.writerow(long_k)
            for vs in long_v:
                wr.writerow(vs)
        myfile.close()

    def export_data_as_csv(self, filename="data", save=True):
        self.create_output_folder(save=save)
        path = self.model.output_path

        with open(path + filename + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(short_k)
            wr.writerow(short_v)
            wr.writerow(long_k)
            for vs in long_v:
                wr.writerow(vs)
        myfile.close()
        pass

    def csv_into_columns(self, input_csv, filename="results_as_col"):  # TODO : doesn't work..
        path = self.model.output_path
        with open(input_csv) as inf:
            with open(path + filename + '.csv', 'w') as outf:
                for line in inf:
                    outf.write('\\'.join(line.split(',')))

    def export_results_as_text(self, filename="results", save=True):
        res_dict = self.displayed_results.dictionary_of_results()
        self.create_output_folder(save=save)
        path = self.model.output_path
        fo = open(path + filename + ".txt", 'w')
        for k, v in res_dict.items():
            if len(res_dict[k]) <= 1:
                fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
            if len(res_dict[k]) > 1:
                fo.write(str(k) + ' >>> ' + '\n\n')
                for i in range(len(res_dict[k])):
                    fo.write(str(i) + '  ' + str(res_dict[k][i]) + '\n\n')
        fo.close()

    def export_results_as_text_adv(self, key_a = "flat_powers", key_b = "theta_opt", filename="results_adv", save=True): #TODO: not working
        res_dict = self.displayed_results.dictionary_of_results()
        self.create_output_folder(save=save)
        path = self.model.output_path
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

    def export_results_as_json(self, filename="results", save=True):  #TODO: not working
        res_dict = self.displayed_results.dictionary_of_results()
        self.create_output_folder(save=save)
        path = self.model.output_path
        import json
        with open(path + filename +"_json_fmt" + ".csv", 'w') as file:
            file.write(json.dumps(res_dict))
        file.close()


