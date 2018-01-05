import pprint
import time
import csv

TEACHER_TO_STUDENT_RATIO = 5

TBP = 'LP2 - Classic True Beginner - Portuguese'
TBS = 'LP2 - Classic True Beginner - Spanish'
INT = 'LP2 - Classic Intermediate'

pp = pprint.PrettyPrinter(indent=4)
f =  open('previous_last_hours.csv', 'r')
reader =  csv.reader(f, delimiter=',')
previous = []

#next(reader)
for row in reader:
        previous.append(row)


print("previous_before")
pp.pprint( previous)



first_block = time.strptime("00:00:00", "%H:%M:%S")
second_block = time.strptime("00:30:00", "%H:%M:%S")


def is_course_ok(prev_shift, curr_type, curr_start): #checks if the shift from prev week could be used to substract a request at a timeblock
        if curr_type == "Private" :
                return True
        elif curr_type == "Live" and get_timediff( prev_time, curr_start )  <= 60:
                return True
        else:
                return False
                
  

def is_lang_ok(prev_shift, curr_level, curr_lang): #checks if the lang of the shift from previous week could be used to substract a request at a timeblock
        if prev_shift == TBP and (curr_level != 'True Beginner' or curr_lang == 'Portuguese'):
                return True
        if prev_shift == TBS and (curr_level != 'True Beginner' or curr_lang == 'Spanish'):
                return True
        if prev_shift == INT and curr_level != 'True Beginner' :
                return True
        else:
               return False

def can_substract(prev_shift, prev_start, prev_end, curr_start, curr_type, curr_lang):
        return is_lang_ok(prev_shift, curr_level, curr_lang) and is_course_ok(prev_shift, curr_type, curr_start)

def datetime_to_time(datetime_str):
        return time.strptime(datetime_str.split()[1].split(".")[0], "%H:%M:%S")


def get_quantity (course_type):
        if course_type == 'Private':
                return 1
        else:
                return TEACHER_TO_STUDENT_RATIO
        
def get_timediff(prev_time, curr_time):
        d1 = datetime_to_time(prev_time)
        d2 = datetime_to_time(curr_time)
        td = d1-d2
        return td.total_seconds() // 60
        
current =[
 ['2018-01-01 00:00:00.000','Private','Portuguese', 'True Beginner',5],
 ['2018-01-01 00:30:00.000','Private','Portuguese', 'True Beginner',0],
 ['2018-01-01 01:00:00.000','Private','Portuguese', 'True Beginner',0],
 ['2018-01-01 00:00:00.000','Private','Spanish', 'True Beginner',0],
 ['2018-01-01 00:30:00.000','Private','Spanish', 'True Beginner',0],
 ['2018-01-01 01:00:00.000','Private','Spanish', 'True Beginner',0],
 ['2018-01-01 00:00:00.000','Private','Spanish', 'True Beginner',0],
 ['2018-01-01 00:30:00.000','Private','Spanish', 'True Beginner',0],
 ['2018-01-01 01:00:00.000','Private','Spanish', 'True Beginner',0]
 ]

print("current_before")
pp.pprint( current)

i=1 #exclude the dates from the loop
while i < len(previous):
        prev_shift = previous[i][2]
        prev_start = previous[i][0]
        prev_end = previous[i][1]
        prev_q = int(previous[i][3])
        j = 0 # current_counter
        while prev_q > 0 and j < len(current) -1 :
                
                while j < len(current) -1 :
                        curr_start = current[j][0]
                        curr_type = current[j][1]
                        curr_lang = current[j][2]
                        curr_level = current[j][3]
                        needed_requests = int(current[j][4])
                        print("-")
                        print(prev_q)
                       
                        print(needed_requests)
                        print("-")
                        while can_substract( prev_shift, prev_start, prev_end, curr_start, curr_type, curr_lang) \
                        and needed_requests > 0 and prev_q > 0:
                                needed_requests = needed_requests - get_quantity(curr_type)
                                current[j][4] = needed_requests 
                                prev_q = prev_q - 1
                                previous[i][3] = prev_q

                        j += 1       

        i += 1
        
print("current_after")
pp.pprint( current)
print("previous")
pp.pprint( previous)



