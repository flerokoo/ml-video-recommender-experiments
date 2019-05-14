def get_num_of_lines(file, include_empty=False):
    f = open(file, "r")
    line = f.readline()
    n = 0
    while line:
        if len(line.strip(" \r\n   ")) > 0 or include_empty:
            n += 1
        line = f.readline()
    return n
