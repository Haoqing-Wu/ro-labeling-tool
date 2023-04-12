import argparse

from mat_loader import MatLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    parser.add_argument('--data_folder', default='./data/', help='path of the mat file which will be processed')
    parser.add_argument('--logs_folder', default='./labels/',
                        help='output path to the folder where labeled training file will be saved.')
    parser.add_argument('--range', default=4.0,
                        help='range of the actor trajectory which will be processed [s].')
    parser.add_argument('--start_frame', default=0,
                        help='the start frame in the recording of the labeling process')
    args = parser.parse_args()
    mat_loader = MatLoader(args)
    mat_loader.generate_ego_paths()
    mat_loader.generate_actors_paths()

