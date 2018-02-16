import pprint
import time
import csv
from dateutil.relativedelta import relativedelta
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO) #set to debug to troubleshoot
logger = logging.getLogger(__name__)


#### This script loops from the forecasted requests output from the model
#### and gets the previous weeks shifts, substracting the number of request from the forecast
#### based on the type of shift, level_category and class_date of the previous week shifts


TEACHER_TO_STUDENT_RATIO = 5
SEP='---------------------------------'

TBP = 'LP2 - Classic True Beginner - Portuguese'
TBS = 'LP2 - Classic True Beginner - Spanish'
INT = 'LP2 - Classic Intermediate'

pp = pprint.PrettyPrinter(indent=4)
f_prev_shifts =  open('previous_last_hours.csv', 'r') # comes from DB
f_forecasted_req =  open('forecasted_requests.csv', 'r') #comes from forecast model
f_final_result =  open('final.csv', 'w') # output

previous_shifts_reader =  csv.reader(f_prev_shifts, delimiter=',')
current_reader = csv.reader(f_forecasted_req, delimiter=',')
final_writer = csv.writer(f_final_result, delimiter=',', lineterminator='\n' )

def to_datetime_from_db(datetime_str): #handles date formatting drom DB input
        
        return datetime.strptime(datetime_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
        #return time.strptime(datetime_str.split()[1].split(".")[0], "%H:%M:%S")

def to_datetime_from_forecast(datetime_str): #handles date formatting from forecast input
        
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")     

current_requests = []
for i, row in enumerate(current_reader):
        
        if i == 0:
                # class_date, level_category, language, course_type, q_predicted
                tmp=[row[0], row[5], row[6], row[7], row[8]]            
        else:
                tmp = [to_datetime_from_forecast(row[0]).strftime ("%Y-%m-%d %H:%M:%S"), row[5], row[6], row[7], row[8].split(".")[0]]
        current_requests.append(tmp)

previous_shifts = []

for row in previous_shifts_reader:
        previous_shifts.append(row)


for i, row in enumerate(previous_shifts_reader):
        if i == 0:
                tmp=row
        else:
                tmp = [to_datetime_from_db(row[0]).strftime ("%Y-%m-%d %H:%M:%S"), row[1], row[2], row[3], row[4]]
        previous_shifts.append(tmp)
        
logger.debug("from forecast:")
logger.debug( current_requests[1:2])
              
logger.debug("previous_before")
logger.debug( previous_shifts[1:2])


#pp.pprint(previous)

#checks if the shift from prev week could be used to substract a request at a timeblock
def is_course_ok(prev_shift, prev_start, curr_type, curr_start): 
        logger.debug("checking course type timing... ")
        if curr_type == "Private" and get_timediff( prev_start, curr_start ) <= 90:
                logger.debug(f"type: {curr_type} prev: {prev_start} curr: {curr_start}  **OK**")
                return True
        elif curr_type == "Live" and get_timediff( prev_start, curr_start )  <= 60:
                logger.debug(f"type: {curr_type} prev: {prev_start} curr: {curr_start}  **OK**")
                return True
        else:
                logger.debug(f"type: {curr_type} prev: {prev_start} curr: {curr_start} **FAIL**")
                return False
                 
#checks if the lang of the shift from previous week could be used to substract a request at a timeblock
def is_lang_ok(prev_shift, curr_level, curr_lang): 
        logger.debug("checking level and language...")
        if prev_shift == TBP and (curr_level != 'True Beginner' or curr_lang == 'Portuguese'):
                logger.debug(f"prev: {prev_shift} curr_level: {curr_level}, curr_lang: {curr_lang} **OK**")
                return True
        if prev_shift == TBS and (curr_level != 'True Beginner' or curr_lang == 'Spanish'):
                logger.debug(f"prev: {prev_shift} curr_level: {curr_level}, curr_lang: {curr_lang} **OK**")
                return True
        if prev_shift == INT and curr_level != 'True Beginner' :
                logger.debug(f"prev: {prev_shift} curr_level: {curr_level}, curr_lang: {curr_lang} **FAIL**")
                return True
        else:
               logger.debug(f"prev: {prev_shift} curr_level: {curr_level}, curr_lang: {curr_lang} **FAIL**")
               return False

def can_substract(prev_shift, prev_start, curr_start, curr_type, curr_lang):
        return is_lang_ok(prev_shift, curr_level, curr_lang) and is_course_ok(prev_shift, prev_start, curr_type, curr_start)

def get_quantity (course_type):
        if course_type == 'Private':
                return 1
        else:
                return TEACHER_TO_STUDENT_RATIO
        
def get_timediff(prev_time, curr_time):
        d1 = to_datetime_from_db(prev_time)
        d2 = to_datetime_from_db(curr_time)
 
        t_diff = relativedelta(d2, d1)
   
        return t_diff.hours * 60 + t_diff.minutes

def get_daysdiff(prev_time, curr_time):
        d1 = to_datetime_from_db(prev_time)
        d2 = to_datetime_from_db(curr_time)
 
        t_diff = relativedelta(d2, d1)
   
        return t_diff.days                            

i=1 #exclude the headers from the loop
while i < len(previous_shifts):
        prev_start = previous_shifts[i][0]
        prev_shift = previous_shifts[i][2]
        prev_q = int(previous_shifts[i][3])
        
        j = 1 # current_shifts_counter
        logger.debug( f"looping forecast row {i}:" )
        while prev_q > 0 and j < len(current_requests) -1 :
                
                while j < len(current_requests) -1 and prev_q > 0:
                        curr_start = current_requests[j][0]
                        curr_level = current_requests[j][1]
                        curr_lang = current_requests[j][2]
                        curr_type = current_requests[j][3]
                        needed_requests = int(current_requests[j][4])
                       
                        time_ = time.strptime(curr_start.split(" ")[1], "%H:%M:%S")
                        
                        if time_.tm_hour >= 2 or get_daysdiff( prev_start, curr_start) > 1 : # exclulde later hours
                                j += 1
                                continue
                        
                        logger.debug(SEP)
                        logger.debug(f"got {prev_q} shift/s {prev_shift} at: \n {prev_start}")
                                               
                        logger.debug(f"\n and in {curr_start}: got {needed_requests} requests \n type: {curr_type} \n level: {curr_level}")
                        
                        while needed_requests > 0 and prev_q > 0 \
                        and can_substract( prev_shift, prev_start,  curr_start, curr_type, curr_lang):
                                q_to_substract = get_quantity(curr_type)
                                logger.debug(f"substracted {q_to_substract} request")
                                
                                needed_requests = max(needed_requests - q_to_substract,0)
                                
                                logger.debug(f"new needed requests ->  {needed_requests}" )
                                current_requests[j][4] = needed_requests 
                                prev_q = prev_q - 1
                                previous_shifts[i][3] = prev_q
                        logger.debug(SEP)
                        j += 1       
        i += 1
        

final_writer.writerows(current_requests)

f_prev_shifts.close()
f_forecasted_req.close()
f_final_result.close()

#pp.pprint(previous)
