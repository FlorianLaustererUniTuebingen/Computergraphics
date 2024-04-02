def mask_number_words(file_name):
    number_words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "no": 11}
    
    with open(file_name, 'r') as f:
        lines = f.readlines()

    masked_lines = []
    for line in lines:
        for word, number in number_words.items():
            if word in line:
                new_line = line.replace(f" {word} ", ' <mask> ', 1)
                if new_line == line:
                    continue
                new_line = new_line.rstrip() + '\t' + word
                masked_lines.append(new_line)

    with open('masked_'+file_name, 'w') as f:
        for line in masked_lines:
            f.write("%s\n" % line)

# Call the function with your file name
mask_number_words('gkb_best_filtered.txt')
