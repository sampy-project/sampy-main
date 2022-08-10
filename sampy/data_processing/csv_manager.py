import numpy as np


class ParamManager:
    def __init__(self, names, values):
        for name, value in zip(names, values):
            setattr(self, name, value)


class CsvManager:
    """
    Realistically, most of Sampy's run will be done during sensitivity analysis to assert that the model is a faithful
    representation of the ecological system of interest. The current class is Sampy's solution for dealing with vast CSV
    of parameters used to run large scale sensitivity analysis.

    The expected CSV structure is as follows.

        - Each row should correspond to the parameters used in a single run of the model.
        - Each column corresponds either to a constant parameter, or to a single value within an array.
        - the current class distinguishes parameters that should be stored in array based on their name in the header.
          That is, if a column name is of the form arr_[some name]_[some_number], then the content of this column will
          be considered as the [some_number]-th element of an array. The array's name will be [some_name].

    Let show the use of CsvManager class on a small example. Assume we have a csv of parameter at the adress path_csv,
    and that the two first line of the csv are:

    const1;const2;arr_test_array_0;arr_test_array_1;arr_another_array_0
    0;wonderful_string_of_chars;2.;3.;True

    One can instantiates a CsvManager class the following way:

    >>> csv_manager = CsvManager(path_csv, ';', dict_types={'const1': int, 'test_array': float, 'another_array': bool})

    Then, by calling the method 'get_parameters', one gets a ParamManager object whose attributes are the parameters
    stored in a line of the csv.

    >>> param = csv_manager.get_parameters()
    >>> print(param.test_array)
    array([2., 3.])

    If one calls get_parameters another time, it will return another ParamManager object corresponding to the next line
    in the csv. Once the end of the csv is reached, get_parameters returns None.

    The kwargs 'nb_cores' and 'id_process' are designed for large analysis using multiple cores. If used, the obtained
    csv_manager will only return lines 'i' in the csv such that 'i % nb_cores == id_process'.

    Finally, when working on very large csv one can use the kwarg buffer_size (default 1000) which says how many
    lines of the csv are stored in memory (CsvManager does not try to open the file entirely in memory, and process it
    by blocks of buffer_size lines).
    """
    def __init__(self, path_to_csv, sep, dict_types=None, buffer_size=1000, nb_cores=1, id_process=0):
        self.path_to_csv = path_to_csv
        self.sep = sep
        if dict_types is None:
            self.dict_types = {}
        else:
            self.dict_types = dict_types

        self.buffer_size = buffer_size
        self.nb_line_consumed = 0
        self.buffer = []
        self.counter_buffer = 0

        self.nb_usable_lines_in_csv = 0

        self.dict_arr = {}
        self.dict_const = {}

        self.nb_cores = nb_cores
        self.id_process = id_process

        with open(self.path_to_csv, 'r') as f_in:
            for i, line in enumerate(f_in):
                if i == 0:
                    self.header = line.replace('\n', '')
                    self.extract_info_header()
                    continue
                if i % self.nb_cores == self.id_process:
                    self.nb_usable_lines_in_csv += 1

    def extract_info_header(self):
        list_header = self.header.split(self.sep)
        dict_col_to_index = {col_name: ind for ind, col_name in enumerate(list_header)}
        r_dict_const = {}
        temp_dict_arr = {}
        for col_name in list_header:
            if col_name.split('_')[0] == 'arr':
                name_param = '_'.join(col_name.split('_')[1:-1])
                try:
                    temp_dict_arr[name_param].append(col_name)
                except KeyError:
                    temp_dict_arr[name_param] = [col_name]
            else:
                r_dict_const[col_name] = dict_col_to_index[col_name]
        r_dict_arr = {}
        for name_arr, arr in temp_dict_arr.items():
            sorted_arr = sorted(arr, key=lambda y: int(y.split('_')[-1]))
            r_dict_arr[name_arr] = [dict_col_to_index[name_col] for name_col in sorted_arr]
        self.dict_arr = r_dict_arr
        self.dict_const = r_dict_const

    def get_parameters(self):
        try:
            line = self.buffer[self.counter_buffer]
            self.counter_buffer += 1
            self.nb_line_consumed += 1
        except IndexError:
            if self.nb_line_consumed == self.nb_usable_lines_in_csv:
                return
            self.fill_buffer()
            line = self.buffer[0]
            self.counter_buffer = 1
            self.nb_line_consumed += 1
        return self.create_param_manager_from_line(line)

    def fill_buffer(self):
        self.buffer = []
        size_current_buffer = 0
        with open(self.path_to_csv) as f:
            seen_lines = 0
            for i, line in enumerate(f):
                if i == 0:
                    continue
                if i % self.nb_cores == self.id_process:
                    seen_lines += 1
                    if seen_lines <= self.nb_line_consumed:
                        continue
                    self.buffer.append(line.replace('\n', ''))
                    size_current_buffer += 1
                    if size_current_buffer == self.buffer_size:
                        break
        return

    def create_param_manager_from_line(self, line):
        data = line.split(self.sep)
        names = []
        values = []
        for name in self.dict_const:
            names.append(name)
            if name in self.dict_types:
                if self.dict_types[name] == bool:
                    values.append(data[self.dict_const[name]].lower() == 'true')
                else:
                    values.append(self.dict_types[name](data[self.dict_const[name]]))
            else:
                values.append(data[self.dict_const[name]])
        for name in self.dict_arr:
            names.append(name)
            if name in self.dict_types:
                if self.dict_types[name] == bool:
                    values.append(np.array([data[u].lower() == 'true' for u in self.dict_arr[name]]))
                else:
                    values.append(np.array([self.dict_types[name](data[u]) for u in self.dict_arr[name]]))
            else:
                values.append(np.array([data[u] for u in self.dict_arr[name]]))
        return ParamManager(names, values)



