import csv



temp = {}
teacherName = []

with open('output.csv', 'r') as f:
  reader = csv.reader(f)
  csv_input = list(reader)
  
temp = {}
pos = []

for row in csv_input:
    pos.append(row)


#print (",".join(calendar))
##print (pos[0][1])
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
print(pos)
while i < len(pos):

    j=0
    while j < len(pos[i]):
        if  j == 0 and pos[i][j] != 'off':
            dist = len(pos[i]) - j -1
            if dist >= 3:
                to_set = 3
            else:
                to_set = dist
            k = 1
            starts[i][j] = pos[i][j]
            while k <= to_set:
               
                starts[i][j+k] = 'o'
                k += 1
    
           

        elif j > 0 and (starts[i][j-1] == '-' or starts[i][j] == '0')  and pos[i][j] != 'off' :
            starts[i][j] = pos[i][j]

            dist = len(pos[i]) - j -1
            print('dist', dist)
            if dist >= 3:
                to_set = 3
            else:
                to_set = dist
            k = 1
            starts[i][j] = pos[i][j]
            while k <= to_set:
                print('k', k)
                starts[i][j+k] = 'o'
                k += 1
            
    
                
            
                       
   
        j +=1    
        
    i+=1
    
print()
print(starts)



def get_shift_type (blocks):
    tbp = {'pp','lp'}
    tbs = {'ps','ls'}
    sets = set(blocks)
    if tbp & sets == set():
        answer = 'tbs'
    elif tbs & sets == set():
        answer = 'tbp'
    elif tbs & tbs & sets != set():
        answer = 'err'
    else:
        answer = 'cla'

    return answer
    



