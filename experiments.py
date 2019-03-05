
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
    value_distance = ['value', 'quarter', 'closest', 'sqrt', 'normal']
    visualize = False

    try:
        print('Started exploring\n')
        with open('experiments_all_cumulative_score.csv', mode='w') as experiments:
            experiment_writer = csv.writer(experiments, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(1, num_tunnel_files + 1):
                # print('')
                print("##################")
                print("Tunnel {}".format(i))
                tunnel_file = './maps/tunnel_{}.npy'.format(i)
                artifact_file = './maps/artifacts_{}.npy'.format(i)
                for e in value_distance:
                    print("Value", e)
                    steps, reward, score_list, points_found = testing.main(e, tunnel_file, artifact_file, visualize)
                    # print("Steps", steps)
                    # print("Reward", reward)
                    # print("POIs found", len(points_found))
                    to_write = concatenate([['tunnel_{}'.format(i)], ['method_{}'.format(e)], [steps], [reward], [sum(score_list)], score_list])
                    experiment_writer.writerow(to_write)

    except (KeyboardInterrupt, SystemExit):
        raise

