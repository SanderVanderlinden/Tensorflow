import sys
import os


def split(input_file, training_data_file, test_data_file, amount_of_lines):


    file = open(input_file)
    with open(training_data_file, 'w') as training:
            with open(test_data_file, 'w') as test:
                i = 0
                for line in file:
                    if i < amount_of_lines:
                        training.write(line)
                    else:
                        test.write(line)
                    i += 1
    file.close()

dirpath = os.getcwd()
input_file = dirpath + '/' + sys.argv[1]
amount_of_lines = int(sys.argv[2])

split(input_file, dirpath + "/nlcow_training.txt", dirpath + "/nlcow_test.txt", amount_of_lines)