import pprint
import time
import csv


pp = pprint.PrettyPrinter(indent=4)
f =  open('previous_last_hours.csv', 'r')
reader =  csv.reader(f, delimiter=',')
_previous = []

next(reader)
for row in reader:
        _previous.append(row)
        #_previous.append(list(map(int, row[1:])))


print("_previous")
pp.pprint( _previous)
previous = []

i =0
while i < len(_previous[0]):
        row = []
        j = 0
        while j < len(_previous[0])-1:
                row.append(_previous[j][i])
                j += 1
        
        previous.append(row)
        i += 1

print("previous")
pp.pprint( previous)

first_block = time.strptime("00:00:00", "%H:%M:%S")
second_block = time.strptime("00:30:00", "%H:%M:%S")

def is_time_ok (previous_col, current_time): #checks if the shift from the previous week can be used in a timeblock
        if previous_col == 2 \
           or \
           (previous_col == 1 and current_time <= second_block) or (previous_col == 0 and current_time <= first_block):
                return True
        else:
                return False

def is_course_ok(previous_column, current_course): #checks if the shift from prev week could be used to substract a request at a timeblock
        if current_course == "Private":
                return True
        elif previous_column < 2:
                return True
        else:
                return False
                
        return True

def is_lang_ok(previous_row, current_level, current_lang): #checks if the lang of the shift from previous week could be used to substract a request at a timeblock

        if previous_row == 0 and (current_lang == "Portuguese" or (current_level == "Intermediate" or current_level == "High Beginner")): 
                return True
        elif previous_row == 1 and (current_lang == "Spanish" or (current_level == "Intermediate" or current_level == "High Beginner")):
                return True
        elif previous_row == 2 and (current_level == "Intermediate" or current_level == "High Beginner"):
                return True
        else:
                return False


def datetime_to_time(datetime_str):
        return time.strptime(datetime_str.split()[1].split(".")[0], "%H:%M:%S")


def get_quantity (course_type):
        if course_type == 'Private':
                return 1
        else:
                return 8
        
                
        
current =[
 ['2018-01-08 00:00:00.000','Private','Portuguese', 'True Beginner',0],
 ['2018-01-08 00:30:00.000','Private','Portuguese', 'True Beginner',0],
 ['2018-01-08 01:00:00.000','Private','Portuguese', 'True Beginner',1],
 ['2018-01-08 00:00:00.000','Private','Spanish', 'True Beginner',0],
 ['2018-01-08 00:30:00.000','Private','Spanish', 'True Beginner',1],
 ['2018-01-08 01:00:00.000','Private','Spanish', 'True Beginner',1],
 ['2018-01-08 00:00:00.000','Private','Spanish', 'True Beginner',1],
 ['2018-01-08 00:30:00.000','Private','Spanish', 'True Beginner',1],
 ['2018-01-08 01:00:00.000','Private','Spanish', 'True Beginner',1]
 ]

i=1 #exclude the dates from the loop
while i < len(previous):
        j=0
        while j < len(previous[i]):
                shifts_available = int(previous[i][j])
                k = 0
                if shifts_available > 0:
                        while k < len(current):
                                needed_requests = current[k][4]
                                q = get_quantity(current[k][1])
                                while needed_requests - q >= 0 and shifts_available > 0 \
                                      and is_time_ok(j, datetime_to_time(current[k][0])) \
                                      and is_course_ok(j, current[k][1]) \
                                      and is_lang_ok(i, current[k][3], current[k][2]):
                                        needed_requests = needed_requests -q
                                        shifts_available = shifts_available -1
                                                           
                                        current[k][4] = needed_requests
                                        previous[i][j] = shifts_available
                                        
                                k+=1
                j +=1
        i+=1
print("current")
pp.pprint( current)
print("previous")
pp.pprint( previous)
print(is_time_ok(0, datetime_to_time("2018-01-08 00:30:00.000")))
print(is_course_ok(0,"Live"))


