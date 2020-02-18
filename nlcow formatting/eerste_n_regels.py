def shorten(input_file, output_file, amount_of_lines):

    with open(input_file, 'r', errors='ignore') as file:
        contents = [next(file) for i in range(amount_of_lines)]

    with open(output_file, 'w') as file:
        for i in range(amount_of_lines):
            file.write(contents[i])


input_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/gitmap/Tensorflow/nlcow formattinga/nlcow.tok"
output_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/gitmap/Tensorflow/nlcow formatting/nlcow10k.tok"
amount_of_lines = 10000

shorten(input_file, output_file, amount_of_lines)