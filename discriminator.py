import argparse

from mat_loader import MatLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    parser.add_argument('--data_folder', default='./data/', help='path of the mat file which will be processed')
    parser.add_argument('--logs_folder', default='./labels/',
                        help='output path to the folder where labeled training file will be saved.')

    args = parser.parse_args()
    MatLoader(args)

