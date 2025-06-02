import os
import json

if __name__ == '__main__': 

	looper_outputs_dir = './exhaustive_search/looper_outputs'
	looper_outputs_paths = os.listdir(looper_outputs_dir) 

	logfiles_paths = [f'{looper_outputs_dir}/{path}/USE_CASE_1/decisions_log.json' for path in looper_outputs_paths]

	# open starting JSON file
	with open(logfiles_paths[0], 'r') as file:
	    basic_config_file = json.load(file)
	print(basic_config_file)

