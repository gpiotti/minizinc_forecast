import time
from datetime import datetime, timedelta
import sys

f_input =  open('minizinc_input.csv', 'r')
f_output = open('data.dzn', 'w')
requests = f_input.read()
FIXED = 'maxTime = 336; \n'

start_date = datetime.strptime(sys.argv[1], "%Y%m%d") 
dates = []

for i in range(0,336):
    dates.append("\"" + start_date.strftime ("%Y-%m-%d %H:%M:%S") + "\"")
    start_date = start_date + timedelta(minutes=30)
    


END_BRACKET = '];\n'

MIN_NEEDED_HEAD = 'minNeeded = array2d(real_blocks, TIME, [ \n'

TIME_LABELS_HEAD = 'timeLabels = \n'


f_output.write(FIXED)
f_output.write(MIN_NEEDED_HEAD)
f_output.write(requests)
f_output.write(END_BRACKET)
f_output.write(TIME_LABELS_HEAD)
f_output.write(",".join(dates))
f_output.write(END_BRACKET)


f_input.close()
f_output.close()
