
# Run testing.py
# Return location of each POI found

import testing_NN
import csv
from numpy import concatenate


def shutdown():
    print('\nGoodbye')


if __name__ == "__main__":

    grid_size = 16 # 16
    num_tunnel_files = 100
    value_distance = ['value', 'quarter', 'closest', 'sqrt', 'normal']
    visualize = False

    try:
        print('Started exploring\n')
        with open('experiments_all_cumulative_score_{}.csv'.format(grid_size), mode='w') as experiments:
            experiment_writer = csv.writer(experiments, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, num_tunnel_files):
                # print('')
                print("##################")
                print("Tunnel {}".format(i))
                tunnel_file = './maps_{}/tunnel_{}.npy'.format(grid_size, i)
                artifact_file = './maps_{}/artifacts_{}.npy'.format(grid_size, i)
                for e in value_distance:
                    print("Value", e)
                    steps, reward, score_list, points_found = testing_NN.main(e, tunnel_file, artifact_file, visualize)
                    # print("Steps", steps)
                    # print("Reward", reward)
                    # print("POIs found", len(points_found))
                    to_write = concatenate([['tunnel_{}'.format(i)], ['method_{}'.format(e)], [steps], [reward], [sum(score_list)], score_list])
                    experiment_writer.writerow(to_write)

    except (KeyboardInterrupt, SystemExit):
        raise

