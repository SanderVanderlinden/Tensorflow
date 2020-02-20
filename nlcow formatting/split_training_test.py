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


input_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/gitmap/Tensorflow/nlcow formatting/nlcow10.tok"
training_data_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/gitmap/Tensorflow/nlcow formatting/nlcow10_training.tok"
test_data_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/gitmap/Tensorflow/nlcow formatting/nlcow10_validation.tok"
amount_of_lines = 4

split(input_file, training_data_file, test_data_file, amount_of_lines)