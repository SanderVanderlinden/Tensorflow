def shorten(input_file, output_file, amount_of_lines):

    with open(input_file, 'r', errors='ignore') as file:
        contents = [next(file) for i in range(amount_of_lines)]

    with open(output_file, 'w') as file:
        for i in range(amount_of_lines):
            file.write(contents[i])

def take_lines(input_file, output_file, amount_of_lines, start_line):

    with open(input_file, 'r', errors='ignore') as file:
        contents = [next(file) for i in range(start_line + amount_of_lines)]

    with open(output_file, 'w') as file:
        for i in range(amount_of_lines):
            file.write(contents[i + start_line])


input_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/nlcow_pos_unknown_added.txt"
output_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/5k_lijnen.txt"
amount_of_lines = 5000
start_line = 10000

#shorten(input_file, output_file, amount_of_lines)
take_lines(input_file, output_file, amount_of_lines, start_line)