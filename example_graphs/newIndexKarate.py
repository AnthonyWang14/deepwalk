fr = open('karate.adjlist')
fw = open('new_karate.adjlist', 'w')
for line in fr.readlines():
	lineArr = line.strip().split()
	strLine = []
	strLine.append(str(int(lineArr[0])-1))
	for i in range(1, len(lineArr)):
		strLine.append(' '+str(int(lineArr[i])-1))
	fw.writelines(strLine)
	fw.write('\n')