"""
python script that compares processed with raw NC files (pre and post generate)
"""

from netCDF4 import Dataset
from pyfiglet import Figlet
from dateutil.parser import isoparse
from tqdm import tqdm

import calendar
import json
import csv
import numpy as np
from numpy import ma

# To show all data in array, use this
# import sys
# np.set_printoptions(threshold=sys.maxsize)

DEBUG = False

text = Figlet(font='standard', width=150)

data_folder = 'nc_files/'

#                      PROCESSED                     RAW              array_equal or isclose  atol val
key_groups = [('sea_surface_temperature',     'geophysical_data/sst',        'isclose',     '0.0001'),
              ('sea_surface_temperature_4um', 'geophysical_data/sst4',       'isclose',     '0.0001'),  # sst4
              ('quality_level',               'geophysical_data/qual_sst',   'array_equal', '0.0001'),
              ('sses_bias',                   'geophysical_data/bias_sst',   'isclose',     '0.1542'),
              ('sses_bias_4um',               'geophysical_data/bias_sst4',  'isclose',     '0.1542'),  # sst4
              ('sses_standard_deviation',     'geophysical_data/stdv_sst',   'isclose',     '0.1542'),
              ('sses_standard_deviation_4um', 'geophysical_data/stdv_sst4',  'isclose',     '0.1542'),  # sst4
              ('lat',                         'navigation_data/latitude',    'array_equal', '0.0001'),
              ('lon',                         'navigation_data/longitude',   'array_equal', '0.0001'),
              ('time',                        'time_coverage_start',         'array_equal', '0.0001')]

keys_with_celsius = ['geophysical_data/sst', 'geophysical_data/sst4']
keys_with_quality = ['geophysical_data/qual_sst']
keys_without_one_extra_dimension = ['lat', 'lon']

#                  R  P
quality_mapping = {0: 5,
                   1: 4,
                   2: 3,
                   3: 1,
                   4: 0}


def read_csv_to_tuples(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(tuple(row))    # Convert each row to a tuple
    return data


def iso8601_to_unix_timestamp(iso_string):
    dt = isoparse(iso_string)
    unix_timestamp = calendar.timegm(dt.utctimetuple())
    return unix_timestamp


def celsius_to_kelvin(celsius_array):
    return celsius_array + 273.15


def map_with_lookup(x):
    if ma.is_masked(x):
        return ma.masked
    else:
        return quality_mapping.get(x, x)


def quality_masking_raw_to_processed(dataset):
    vectorized_map = np.vectorize(map_with_lookup)
    mapped_data = vectorized_map(dataset.ravel())
    mapped_data = ma.array(mapped_data, mask=dataset.mask.ravel()).reshape(dataset.shape)
    return mapped_data


def get_fields_value_from_raw(dataset_a, dataset_b):
    # Build dict
    output = {}
    for k, v in dataset_a.groups.items():
        for x, y in v.variables.items():
            key_field = f'{k}/{x}'
            if key_field in [raw_set[1] for raw_set in key_groups]:
                if DEBUG:
                    print(text.renderText(key_field))
                    print(f'source: {k}/{x}')
                    print(y[:])
                if key_field in keys_with_celsius:
                    output[key_field] = celsius_to_kelvin(y[:])
                elif key_field in keys_with_quality:
                    # Only the SST's have qual_sst; qual_sst4 is ignored for now
                    output[key_field] = quality_masking_raw_to_processed(y[:])
                else:
                    output[key_field] = y[:]

    for k, v in dataset_b.groups.items():
        for x, y in v.variables.items():
            key_field = f'{k}/{x}'
            if key_field in [raw_set[1] for raw_set in key_groups]:
                if DEBUG:
                    print(text.renderText(key_field))
                    print(f'source: {k}/{x}')
                    print(y[:])
                if key_field in keys_with_celsius:
                    output[key_field] = celsius_to_kelvin(y[:])
                else:
                    output[key_field] = y[:]

    if dataset_a.__dict__.get('time_coverage_start') != dataset_b.__dict__.get('time_coverage_start'):
        print(f">>>> ERROR ON dataset A/B\'s time_coverage_start not matching "
              f"({dataset_a.__dict__.get('time_coverage_start')})"
              f"({dataset_b.__dict__.get('time_coverage_start')})")

    output['time_coverage_start'] = iso8601_to_unix_timestamp(dataset_a.__dict__.get('time_coverage_start'))
    return output


def get_fields_value_from_processed(dataset):
    # Build dict
    output = {}
    for k, v in dataset.variables.items():
        if k in [processed_set[0] for processed_set in key_groups]:
            if DEBUG:
                print(text.renderText(k))
                print(f'source: {k}')
                print(v[:])
            if k not in keys_without_one_extra_dimension:
                output[k] = v[0]
            else:
                output[k] = v[:]
    output['time'] = iso8601_to_unix_timestamp(dataset.__dict__.get('time_coverage_start'))
    return output


def compare_datasets(processed_dataset, raw_dataset):
    return_dict = {}
    for pairs in key_groups:
        set1 = processed_dataset.get(pairs[0])
        set2 = raw_dataset.get(pairs[1])
        if DEBUG:
            print(text.renderText(f'{pairs[0]}\n{pairs[1]}'))
            print(set1)
            print(set2)

        if set1 is not None or set2 is not None:
            if pairs[2] == 'array_equal':
                compare_result = np.array_equal(set1, set2, equal_nan=True)
                return_dict[pairs[0]] = compare_result
            elif pairs[2] == 'isclose':
                compare_result = np.isclose(set1, set2, atol=float(pairs[3]), equal_nan=True)
                return_dict[pairs[0]] = np.mean(compare_result)
            else:
                print('missing comparison method')

    return return_dict


def main():
    file_list_trios = read_csv_to_tuples('nc_files.csv')
    process_report = {}
    for p, r1, r2 in tqdm(file_list_trios):
        val1 = get_fields_value_from_processed(Dataset(f'{data_folder + p}'))
        val2 = get_fields_value_from_raw(Dataset(f'{data_folder + r1}'), Dataset(f'{data_folder + r2}'))
        if val1 and val2 is not None:
            process_report[p] = compare_datasets(val1, val2)

    pretty_json = json.dumps(process_report, indent=4)
    print(pretty_json)


if __name__ == '__main__':
    main()
