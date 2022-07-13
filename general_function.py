""
" Import librairies "
""
import numpy as np
import pandas as pd
import re
import datetime
import matplotlib.pyplot as plt
import time
import pylab
from tensorflow.keras.preprocessing.sequence import pad_sequences
""
" Function to convert string to datetime "
""
def convert(date_time):
    format = '%Y-%m-%d %H:%M:%S.%f'
    datetime_str = datetime.datetime.strptime(date_time, format)
    return datetime_str
""
" Data Frame read "
""
df = pd.read_csv("DATA_copie.csv",  delimiter=";", header=None, on_bad_lines='skip',engine='python')
#df = df[0:15116]# 65  6986
datetimes = df[0]
sensor = df[1]
value = df[2]
activity = df[3]
sensors, values = list(), list()
for i in range(int(len(df))):
    sensors.append(sensor[i])
    values.append(value[i])
print("len_values=", len(values))
""
" Identification of the begin's index for each activity "
""
def begin_activity_index(datetimes, activity_name_begin: str):
    j_begins = []
    for j in range(int(len(datetimes))):
            if activity[j] == activity_name_begin:
                j_begin= j
                j_begins.append(j_begin)
                #print('j_begins=', j_begins)
    return j_begins
""
" Identification of the end's index for each activity "
""
def end_activity_index(datetimes, activity_name_end: str):
    j_ends = []
    for j in range(int(len(datetimes))):
            if activity[j] == activity_name_end:
                j_end= j
                j_ends.append(j_end)
                #print('j_ends=', j_ends)
    return j_ends
""
" Test of two functions "
""
activity_name_begin = 'Sleeping_end'
res = begin_activity_index(datetimes, activity_name_begin)
""
" Identification of the sensors and their values"
""
def identification_sensors_and_values(j_begins, j_ends, activity_name:str):
    l = []
    for i in range(len(j_begins)):  #
        activity_events = [sensors[j] for j in range(j_begins[i], j_ends[i])]
        values_of_events = [float(values[j]) for j in range(j_begins[i], j_ends[i])]

        l.append(np.array(values_of_events))
        res = l #np.array(l)
        #print("values_of_events=", values_of_events)
    return res
""
" Calculation of the number of padded elements"
""
def test(sequences):
    padded = pad_sequences(sequences)
    L = list()
    for i in range(len(padded)):
        for j in range(len(sequences)):
            l = len(padded[i]) - len(sequences[j])
            L.append(l)
    res = L[0:len(padded)]
    return res
""
" Making test for Bed_to_Toilet activity"
""
Bed_to_Toilet_activity_name = 'Bed_to_Toilet'
Bed_to_Toilet_begin = 'Bed_to_Toilet_begin'
Bed_to_Toilet_end = 'Bed_to_Toilet_end'
Bed_to_Toilet_j_begins = begin_activity_index(datetimes, Bed_to_Toilet_begin)
Bed_to_Toilet_j_ends =   end_activity_index(datetimes, Bed_to_Toilet_end)
res_Bed_to_Toilet = identification_sensors_and_values(Bed_to_Toilet_j_begins, Bed_to_Toilet_j_ends, Bed_to_Toilet_activity_name)
list_of_padded_elements_Bed_to_Toilet = test(res_Bed_to_Toilet)
""
" Making test for Sleeping activity"
""
Sleeping_activity_name = 'Sleeping'
Sleeping_begin = 'Sleeping_begin'
Sleeping_end = 'Sleeping_end'
Sleeping_j_begins = begin_activity_index(datetimes, Sleeping_begin)
Sleeping_j_ends =   end_activity_index(datetimes, Sleeping_end)
res_Sleeping = identification_sensors_and_values(Sleeping_j_begins, Sleeping_j_ends, Sleeping_activity_name)
list_of_padded_elements_Sleeping = test(res_Sleeping)
""
" Making test for Eating activity"
""
Eating_activity_name = 'Eating'
Eating_begin = 'Eating_begin'
Eating_end = 'Eating_end'
Eating_j_begins = begin_activity_index(datetimes, Eating_begin)
Eating_j_ends =   end_activity_index(datetimes, Eating_end)
res_Eating = identification_sensors_and_values(Eating_j_begins, Eating_j_ends, Eating_activity_name)
list_of_padded_elements_Eating = test(res_Eating)
print("list_of_padded_elements_Eating=", list_of_padded_elements_Eating)
""
" Making test for Housekeeping activity "
""
Housekeeping_activity_name = 'Housekeeping'
Housekeeping_begin = 'Housekeeping_begin'
Housekeeping_end = 'Housekeeping_end'
Housekeeping_j_begins = begin_activity_index(datetimes, Housekeeping_begin)
Housekeeping_j_ends =   end_activity_index(datetimes, Housekeeping_end)
res_Housekeeping = identification_sensors_and_values(Housekeeping_j_begins, Housekeeping_j_ends, Housekeeping_activity_name)
list_of_padded_elements_Housekeeping= test(res_Housekeeping)
""
" Making test for Meal_Preparation activity "
""
Meal_Preparation_activity_name = 'Meal_Preparation'
Meal_Preparation_begin = 'Meal_Preparation_begin'
Meal_Preparation_end = 'Meal_Preparation_end'
Meal_Preparation_j_begins = begin_activity_index(datetimes, Meal_Preparation_begin)
Meal_Preparation_j_ends =   end_activity_index(datetimes, Meal_Preparation_end)
res_Meal_Preparation = identification_sensors_and_values(Meal_Preparation_j_begins, Meal_Preparation_j_ends, Meal_Preparation_activity_name)
list_of_padded_elements_Meal_Preparation = test(res_Meal_Preparation)
""
" Making test for Relax activity "
""
Relax_activity_name = 'Relax'
Relax_begin = 'Relax_begin'
Relax_end = 'Relax_end'
Relax_j_begins = begin_activity_index(datetimes, Relax_begin)
Relax_j_ends =   end_activity_index(datetimes, Relax_end)
res_Relax = identification_sensors_and_values(Relax_j_begins, Relax_j_ends, Relax_activity_name)
list_of_padded_elements_Relax = test(res_Relax)
""
" Making test for Work activity "
""
Work_activity_name = 'Work'
Work_begin = 'Work_begin'
Work_end = 'Work_end'
Work_j_begins = begin_activity_index(datetimes, Work_begin)
Work_j_ends =   end_activity_index(datetimes, Work_end)
res_Work = identification_sensors_and_values(Work_j_begins, Work_j_ends, Work_activity_name)
list_of_padded_elements_Work= test(res_Work)
""
" Making test for Enter_Home activity "
""
Enter_Home_activity_name = 'Enter_Home'
Enter_Home_begin = 'Enter_Home_begin'
Enter_Home_end = 'Enter_Home_end'
Enter_Home_j_begins = begin_activity_index(datetimes, Enter_Home_begin)
Enter_Home_j_ends =   end_activity_index(datetimes, Enter_Home_end)
res_Enter_Home = identification_sensors_and_values(Enter_Home_j_begins, Enter_Home_j_ends, Enter_Home_activity_name)
list_of_padded_elements_Enter_Home = test(res_Enter_Home)
""
" Making test for Leave_Home activity "
""
Leave_Home_activity_name = 'Leave_Home'
Leave_Home_begin = 'Leave_Home_begin'
Leave_Home_end = 'Leave_Home_end'
Leave_Home_j_begins = begin_activity_index(datetimes, Leave_Home_begin)
Leave_Home_j_ends =   end_activity_index(datetimes, Leave_Home_end)
res_Leave_Home = identification_sensors_and_values(Leave_Home_j_begins, Leave_Home_j_ends, Leave_Home_activity_name)
list_of_padded_elements_Leave_Home = test(res_Leave_Home)
""
" Plotting "
""
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(list_of_padded_elements_Sleeping, label='padded_elements_Sleeping')
plt.legend()
plt.title('Evolution of the len of padded elements for Sleeping activity')
plt.subplot(3, 1, 2)
plt.plot(list_of_padded_elements_Bed_to_Toilet, label='padded_elements_Bed_to_Toilet')
plt.legend()
plt.title('Evolution of the len of padded elements for Bed_to_Toilet')
plt.subplot(3, 1, 3)
plt.plot(list_of_padded_elements_Housekeeping, label='padded_elements_Housekeeping')
plt.legend()
plt.title('Evolution of the len of padded elements for Housekeeping')
plt.show()

plt.figure(2)
plt.subplot(3, 1, 1)
plt.plot(list_of_padded_elements_Relax, label='padded_elements_Relax')
plt.legend()
plt.title('Evolution of the len of padded elements for Relax')
plt.subplot(3, 1, 2)
plt.plot(list_of_padded_elements_Work, label='padded_elements_Work')
plt.legend()
plt.title('Evolution of the len of padded elements for Work')
plt.subplot(3, 1, 3)
plt.plot(list_of_padded_elements_Meal_Preparation, label='padded_elements_Meal_Preparation')
plt.legend()
plt.title('Evolution of the len of padded elements for Meal_Preparation')
plt.show()

plt.figure(3)
plt.subplot(3, 1, 1)
plt.plot(list_of_padded_elements_Eating, label='padded_elements_Eating')
plt.legend()
plt.title('Evolution of the len of padded elements for Eating')
plt.subplot(3, 1, 2)
plt.plot(list_of_padded_elements_Enter_Home, label='padded_elements_Enter_Home')
plt.legend()
plt.title('Evolution of the len of padded elements for Enter_Home')
plt.subplot(3, 1, 3)
plt.plot(list_of_padded_elements_Leave_Home, label='padded_elements_Leave_Home')
plt.legend()
plt.title('Evolution of the len of padded elements for Leave_Home')
plt.show()