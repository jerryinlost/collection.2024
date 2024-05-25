with open('unique_ids.lst', 'r') as fileA, open('done.lst', 'r') as fileB:
    linesA = set(fileA.readlines())
    linesB = set(fileB.readlines())

result = linesA - linesB

for line in result:
    print(line.strip())