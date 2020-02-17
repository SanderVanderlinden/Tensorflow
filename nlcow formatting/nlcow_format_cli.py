import sys
import os

def tok_to_lem_dict(input_file1, input_file2, output_file):
    contents1 = split_in_words(input_file1)
    contents2 = split_in_words(input_file2)

    output = dict()

    line_index = 0
    for line in contents1:
        print(len(line))
        word_index = 0
        for word in line:
            #print(word_index)
            if word_index == (len(line) - 1):
                None #doe niks
            elif word_index == 0 or word_index == (len(line) - 2):
                None #doe niks
            else:
                output[word] = contents2[line_index][word_index]
            word_index += 1
        line_index += 1

    with open(output_file, 'w') as file:
        #file.write(str(output))
        for word, lemma in output.items():
            file.write(word + ": " + lemma + "\n")


def split_in_words(input_file) :
    with open(input_file, 'r') as file:
        contents = file.read()

    contents = contents.split("\n")

    line_index = 0
    for line in contents:
        contents[line_index] = line.split(" ")
        line_index += 1

    return contents

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

dirpath = os.getcwd()

tok_file = dirpath + '\\' + sys.argv[1]
lem_file = dirpath + '\\' + sys.argv[2]
pos_file = dirpath + '\\' + sys.argv[3]
minimal_occurrences = sys.argv[4]

tok_to_lem_dict(tok_file, lem_file, dirpath + "\\nlcow.toklemdict")
word_count(tok_file, dirpath + "\\nlcow.wc")
filter_uncommon(tok_file, dirpath + "\\nlcow.wc", dirpath + "\\nlcow.toklemdict", dirpath + "\\nlcow.tokfiltered", minimal_occurrences)
combine(dirpath + "\\nlcow.tokfiltered", pos_file, dirpath + "\\nlcow.end_result")

