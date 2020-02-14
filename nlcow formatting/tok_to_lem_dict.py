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
        file.write(str(output))


def split_in_words(input_file) :
    with open(input_file, 'r') as file:
        contents = file.read()

    contents = contents.split("\n")

    line_index = 0
    for line in contents:
        contents[line_index] = line.split(" ")
        line_index += 1

    return contents

input_file1 = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/nlcow10.tok"
input_file2 = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/nlcow10.lem"
output_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/nlcow10.toklemdict"

tok_to_lem_dict(input_file1, input_file2, output_file)