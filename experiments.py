
# Run testing.py
# Return location of each POI found

import testing
import csv
from numpy import concatenate


def shutdown():
    print('\nGoodbye')


if __name__ == "__main__":

    num_runs = 20
    num_tunnel_files = 5
    value_distance = ['value', 'subtract', 'closest', 'sqrt', 'normal']
    visualize = False

    try:
        print('Started exploring\n')
        for i in range(1, num_tunnel_files + 1):
            print('')
            print("##################")
            print("Tunnel {}".format(i))
            tunnel_file = './maps/tunnel_{}.npy'.format(i)
            artifact_file = './maps/artifacts_{}.npy'.format(i)
            with open('experiments_tunnel{}_cumulative_score.csv'.format(i), mode='w') as experiments:
                experiment_writer = csv.writer(experiments, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for e in value_distance:
                    steps, reward, score_list, points_found = testing.main(e, tunnel_file, artifact_file, visualize)
                    print('')
                    # print("Steps", steps)
                    # print("Reward", reward)
                    # print("POIs found", len(points_found))
                    to_write = concatenate([['method_{}'.format(e)], [steps], [reward], score_list])
                    experiment_writer.writerow(to_write)

    except (KeyboardInterrupt, SystemExit):
        raise

