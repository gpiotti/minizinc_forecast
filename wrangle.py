import csv
from datetime import datetime, timedelta
date_str = '2017-11-06 00:00:00'
start_date =  datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
myfile = open('shifts.csv', 'w')
myfile_starts = open('starts.csv', 'w')

def get_shift_type (blocks):
    tbp = {'pp','lp',}
    tbs = {'ps','ls'}
    cla = {'pc','lc'}
    sets = set(blocks)
    if tbp & tbs & sets == set() and cla & sets != set():
        answer = 'cla'
    elif tbp & sets == set():
        answer = 'tbs'
    elif tbs & sets == set():
        answer = 'tbp'
    else:
        answer = 'err'

    return answer

def get_distance(current_d, max_length):
    dist = max_length - current_d -1
    if dist >= 3:
        to_set = 3
    else:
        to_set = dist
    return to_set

with open('output.csv', 'r') as f:
  reader = csv.reader(f)
  csv_input = list(reader)

pos = []

for row in csv_input:
    pos.append(row)


starts = []
i=0
while i < len(pos):
    j=0
    starts.append([])
    while j < len(pos[i]):
        if pos[i][j] != 'off':
            starts[i].append('0')
        else:
            starts[i].append('-')
        j+=1
    i+=1


i=0
while i < len(pos):

    j=0
    while j < len(pos[i]):
        if  j == 0 and pos[i][j] != 'off':
            to_set = get_distance(j, len(pos[i]))             
            k = 1
            starts[i][j] = get_shift_type(pos[i][j:to_set])

            while k <= to_set:
                starts[i][j+k] = 'o'
                k += 1
        elif j > 0 and (starts[i][j-1] == '-' or starts[i][j] == '0')  and pos[i][j] != 'off' :
            to_set = get_distance(j, len(pos[i]))
            k = 1          
            starts[i][j] = get_shift_type(pos[i][j:j+to_set])
 
            while k <= to_set:
                starts[i][j+k] = 'o'
                k += 1                    
        j +=1      
    i+=1
    


final = []

i = 0

while i < len(starts[0]):
    sum_tbp = 0
    sum_tbs = 0
    sum_cla = 0
    j = 0
    while j < len(starts):
        if starts[j][i] == 'tbp': 
            sum_tbp += 1
        elif starts[j][i] == 'tbs':
            sum_tbs += 1
        elif starts[j][i] == 'cla':
            sum_cla += 1
        j += 1
    final.append([sum_tbs, sum_tbp, sum_cla])
    i += 1


i=0
myfile.write('date, tbs, tbp, cla \n')

while i < len(final):
    str_ = (start_date + timedelta(minutes=i*30)).isoformat(sep= " ") + ',' + str(final[i][0]) + ',' + str(final[i][1])     + ',' +  str(final[i][2]) +'\n'
    myfile.write(str_)   
    i += 1

myfile.close()
