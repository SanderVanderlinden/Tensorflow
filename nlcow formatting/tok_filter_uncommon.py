#Haal alle woorden die minder dan n keer voorkomen en een onbekend lemma hebben er uit.

def filter_uncommon(tok_file, wc_file, output_file, n):
    contents1 = split_in_words(tok_file)
    contents2 = make_dict(wc_file)

    output = ""

    line_index = 0
    for line in contents1:
        word_index = 0
        for word in line:
            if word != '':
                if int(contents2[word]) >= int(n):
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

tok_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/nlcow10.tok"
wc_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/nlcow10.wc"
output_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/nlcow10.tokfiltered"
minimal_occurrences = 2

filter_uncommon(tok_file, wc_file, output_file, minimal_occurrences)