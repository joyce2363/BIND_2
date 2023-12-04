import numpy as np
import argparse
import csv 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="input in terminal", help='One dataset from income, bail, pokec1, and pokec2.')
args = parser.parse_args()
dataset_name = args.dataset

final_sets = None
acc_1 = []
sp_1 = []
eo_1 = []

acc_10 = []
sp_10 = []
eo_10 = []
dict = {}
for i in range(1,6): 
    if dataset_name == 'income':
        final_sets = np.load('1final_sets_income' + str(i) + '.npy', allow_pickle=True).item()
    elif dataset_name == 'bail':
        final_sets = np.load('1final_sets_bail' + str(i) + '.npy', allow_pickle=True).item()
    elif dataset_name == 'pokec1':
        final_sets = np.load('1final_sets_pokec1' + str(i) + '.npy', allow_pickle=True).item()
    elif dataset_name == 'pokec2':
        final_sets = np.load('1final_sets_pokec2' + str(i) + '.npy', allow_pickle=True).item()
    elif dataset_name == 'nba': 
        final_sets = np.load('1final_sets_nba' + str(i) + '.npy', allow_pickle=True).item()


    print("BIND 1%:")
    budget = 10
    print("Acc:", final_sets['acc_records'][budget])
    print("FINAL SETS ACC_RECORDS:", final_sets['acc_records'])
    print("LENGTH FINAL SETS ACC_RECORDS:", len(final_sets['acc_records']))
    acc_1.append(final_sets['acc_records'][budget])

    print("Statistical Parity:", final_sets['sp_records'][budget])
    print("FINAL SETS SP_RECORDS:", final_sets['sp_records'])
    print("LENGTH FINAL SETS SP_RECORDS:", len(final_sets['sp_records']))
    sp_1.append(final_sets['sp_records'][budget])
  

    print("Equal Opportunity:", final_sets['eo_records'][budget])
    eo_1.append(final_sets['eo_records'][budget])
    print("FINAL SETS EO_RECORDS:", final_sets['eo_records'])
    print("LENGTH FINAL SETS EO_RECORDS:", len(final_sets['eo_records']))

    print("BIND 10%:")
    budget = 92
    print("Acc:", final_sets['acc_records'][budget])
    acc_10.append(final_sets['acc_records'][budget])
    print("Statistical Parity:", final_sets['sp_records'][budget])
    sp_10.append(final_sets['sp_records'][budget])
    print("Equal Opportunity:", final_sets['eo_records'][budget])
    eo_10.append(final_sets['eo_records'][budget])

print("BIND 1% AVERAGE + VARIANCE:")
# print("variance acc_1:", np.var(acc_1), "+=", )
print("average acc_1:", np.mean(acc_1), "+= ", np.var(acc_1))

# print("variance sp_1:", np.var(sp_1))
print("average sp_1:", np.mean(sp_1), "+= ", np.var(sp_1))

# print("variance eo_1:", np.var(eo_1))
print("average eo_1:", np.mean(eo_1), "+= ", np.var(eo_1))

print("BIND 10% AVERAGE + VARIANCE:")
# print("variance acc_10:", np.var(acc_10))
print("average acc_10:", np.mean(acc_10), "+= ", np.var(acc_10) )

# print("variance sp_10:", np.var(sp_10))
print("average sp_10:", np.mean(sp_10), "+= ", np.var(sp_10) )

# print("variance eo_10:", np.var(eo_10))
print("average eo_10:", np.mean(eo_10), "+= ", np.var(eo_10) )


dict['Model'] = 'BIND 1%'
dict['Dataset'] = str(args.dataset)
dict['Average'] = str(np.mean(acc_1)) + str(" += ") + str(np.var(acc_1))
dict['Statistical Parity'] = str(np.mean(sp_1)) + str(" += ") + str(np.var(sp_1))
dict['Equal Opportunity'] = str(np.mean(eo_1)) + str(" += ") + str(np.var(eo_1))
print(dict)
# Define the filename for the CSV file
filename = 'output.csv'

# Writing data from dictionary to the CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=dict.keys())
    
    # Write header
    writer.writeheader()
    
    # Write rows
    for i in range(0,1):
        row = {key: dict[key] for key in dict}
        writer.writerow(row)

print(f"Data from dictionary has been written to '{filename}' successfully.")