'''
Just a script that combines the different channel combos to create one json, that can be used
for analysis
'''

import json
import csv

with open("analysis/pre-processing/json_base/mcunet-5fps_all.json", 'r') as file:
    mcunet_all = json.load(file)

with open("analysis/pre-processing/json_base/mcunet-5fps_half.json", 'r') as file:
    mcunet_half = json.load(file)

with open("analysis/pre-processing/json_base/mcunet-5fps_quarter.json", 'r') as file:
    mcunet_quarter = json.load(file)

with open("analysis/pre-processing/json_base/mcunet-5fps_eighth.json", 'r') as file:
    mcunet_eighth = json.load(file)

delta_accs_data = []

with open("analysis/pre-processing/analysis_data/flowers_layer_contribution.csv", 'r') as file:
    reader = csv.DictReader(file)
    for line in reader:
        delta_accs_data.append(line)

for i in range(1,43):
    mcunet_all['conv'+str(i)]['half_channels']    = mcunet_half['conv'+str(i)]['channels']
    mcunet_all['conv'+str(i)]['quarter_channels'] = mcunet_quarter['conv'+str(i)]['channels']
    mcunet_all['conv'+str(i)]['eighth_channels']  = mcunet_eighth['conv'+str(i)]['channels']
    if 'weight_count' in mcunet_all['conv'+str(i)]:
        del mcunet_all['conv'+str(i)]['weight_count']
    if 'channels' in mcunet_all['conv'+str(i)]:
        del mcunet_all['conv'+str(i)]['channels']
    mcunet_all['conv'+str(i)]['all_acc_x100'] = int(0)
    mcunet_all['conv'+str(i)]['half_acc_x100'] = int(0)
    mcunet_all['conv'+str(i)]['quarter_acc_x100'] = int(0)
    mcunet_all['conv'+str(i)]['eighth_acc_x100'] = int(0)
    for delta_acc in delta_accs_data:
        if str(i-1)+"_all" == delta_acc['id']:
            mcunet_all['conv'+str(i)]['all_acc_x100'] = int(float(delta_acc['delta acc']) * 100)
        elif  str(i-1)+"_half" == delta_acc['id']:
            mcunet_all['conv'+str(i)]['half_acc_x100'] = int(float(delta_acc['delta acc']) * 100)
        elif  str(i-1)+"_quarter" == delta_acc['id']:
            mcunet_all['conv'+str(i)]['quarter_acc_x100'] = int(float(delta_acc['delta acc']) * 100)

with open("analysis/model_jsons/mcunet-5fps_all.json", 'w') as file:
    json.dump(mcunet_all, file)