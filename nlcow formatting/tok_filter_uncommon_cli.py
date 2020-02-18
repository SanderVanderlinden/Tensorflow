'''
HOW TO USE?
python tok_filter_uncommon_cli.py tok_file wc_file output_file minimal_occurrences
'''

import sys

def filter_uncommon(tok_file, wc_file, toklemdict_file, output_file, n):
    contents1 = split_in_words(tok_file)
    contents2 = make_dict(wc_file)
    contents3 = make_dict(toklemdict_file)

    output = ""

    line_index = 0
    for line in contents1:
        word_index = 0
        for word in line:
            if word != '':
                if int(contents2[word]) >= int(n):
                    output += word
                elif contents3[word] != '<UNK>':
                    output += word
                else:
                    output += "<UNK>"
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

def make_dict(input_file):
    output = dict()

    with open(input_file, 'r') as file:
        contents = file.read()

    contents = contents.split("\n")

    line_index = 0
    for line in contents:
        contents[line_index] = line.split(": ")
        if contents[line_index][0] != '':
            output[contents[line_index][0]] = contents[line_index][1]
        line_index += 1

    return output

filter_uncommon(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
