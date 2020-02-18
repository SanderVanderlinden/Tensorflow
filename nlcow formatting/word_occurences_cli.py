'''
HOW TO USE?
python word_occurences_cli.py tok_file output_file
'''

import sys

def word_count(input_file, output_file):
    contents = split_in_words(input_file)

    output = dict()

    line_index = 0
    for line in contents:
        word_index = 0
        for word in line:
            #print(word_index)
            if word in output:
                output[word] += 1
            else:
                output[word] = 1
            word_index += 1
        line_index += 1

    with open(output_file, 'w') as file:
        #file.write(str(output))
        for word, occurences in output.items():
            file.write(word + ": " + str(occurences) + "\n")

def split_in_words(input_file) :
    with open(input_file, 'r') as file:
        contents = file.read()

    contents = contents.split("\n")

    line_index = 0
    for line in contents:
        contents[line_index] = line.split(" ")
        line_index += 1

    return contents

word_count(sys.argv[1], sys.argv[2])
