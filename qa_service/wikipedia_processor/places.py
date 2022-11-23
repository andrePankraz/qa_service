'''
This file was created by ]init[ AG 2022.

Package for Model "No Language Left Behind" (NLLB).
'''
import logging
import numpy
import pandas


def _convert_places():
    '''
    Only used once for generating CSV with place names.
    Wikipedia dump has not good enough category system to recognize places (cities etc.).
    '''
    data = pandas.read_csv('imports/geodatendeutschland.csv', sep=',', header='infer')
    place_names = sorted(pandas.concat([data['KREIS_NAME'], data['GEMEINDE_NAME'], data['ORT_NAME']]).unique())

    places = pandas.DataFrame(place_names, columns=['name'])
    places.to_csv('imports/places.csv', index=False)


def read_places() -> numpy.ndarray:
    return pandas.read_csv('imports/places.csv').to_numpy()


def main_debug():
    '''
    Just for debugging.
    '''
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # _convert_places()
    places = read_places()
    print('Dresden' in places)


if __name__ == '__main__':
    '''
    Just for debugging.
    '''
    main_debug()
