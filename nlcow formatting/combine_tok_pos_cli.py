'''
HOW TO USE?
python combine_tok_pos_cli.py tok_file pos_file output_file
'''

import sys

def combine(input_file1, input_file2, output_file):
    contents1 = split_in_words(input_file1)
    contents2 = split_in_words(input_file2)

    output = ""

    line_index = 0
    for line in contents1:
        print(len(line))
        word_index = 0
        for word in line:
            #print(word_index)
            if word_index == (len(line) - 1):
                None #doe niks
            elif word_index == 0 or word_index == (len(line) - 2):
                output += word
                print(word_index)
                print(contents1[line_index][word_index])
                print(contents1[line_index][word_index - 1])
            else:
                output += word + "[" + contents2[line_index][word_index] + "]"
            output += " "
            word_index += 1
        output += "\n"
        line_index += 1

    with open(output_file, 'w') as file:
        file.write(output)


def split_in_words(input_file) :
    with open(input_file, 'r') as file:
        contents = file.read()

    contents = contents.split("\n")

    line_index = 0
    for line in contents:
        contents[line_index] = line.split(" ")
        line_index += 1

    return contents

combine(sys.argv[1], sys.argv[2], sys.argv[3])
