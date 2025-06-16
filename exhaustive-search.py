import json
import os
import itertools
import shutil
import numpy as np
from offlineALLclass import AutonomousLooperOffline
from subprocess import Popen # process on multiple threads


if __name__ == '__main__': 

	# define soundfile and basic config
	soundfile_filepath = './exhaustive_search/corpus/USE_CASE_1.wav'
	starting_config_filepath = './exhaustive_search/corpus/objective_config.json'
	
	# open basic JSON config file
	with open(starting_config_filepath, 'r') as file:
	    basic_config_file = json.load(file)
	print(basic_config_file)
	rules = basic_config_file['looping-rules']
	N_RULES = 0
	for rule in rules:
		for rule_component in rule:
			N_RULES += 1

	# generate all parameter permutations
	step = 0.1
	elements = np.arange(0.0, 1.0+step, step)
	permutations = [p for p in itertools.product(elements, repeat=N_RULES)]
	print(permutations[0])
	print(f'num threshold permutations: {len(permutations)}')
	xi_elements = ["less", "more"]
	xi_permutations = [p for p in itertools.product(xi_elements, repeat=N_RULES)]
	print(xi_permutations)
	print(f'num xi permutations: {len(xi_permutations)}')
	print(f'total number of permutations: {len(xi_permutations)*len(permutations)}')

	# generate config files for all rule combinations
	config_files_path = './exhaustive_search/config_files'
	config_file_blank = basic_config_file.copy()
	for p, permutation in enumerate(permutations):
		new_rules = rules.copy()
		for x, xi_permutation in enumerate(xi_permutations):
			i = 0
			for rule in new_rules:
				for rule_component in rule:
					rule_component["rule-threshold"] = permutation[i]
					rule_component["rule-type"] = xi_permutation[i]
					i+=1
			config_file_blank['looping-rules'] = new_rules
			with open(f'{config_files_path}/config_{p}_{x}.json', 'w', encoding='utf-8') as f:
				json.dump(config_file_blank, f, ensure_ascii=False, indent=4)

	# generate commands to run looper for each config file
	config_files_list = os.listdir(config_files_path) 
	command_strings = []
	for config_file in config_files_list:
		config_filepath = f'{config_files_path}/{config_file}'
		output_dir_path = f'./exhaustive_search/looper_outputs/{config_file.split(".")[0]}'
		if os.path.isdir(output_dir_path):
			shutil.rmtree(output_dir_path)
		os.mkdir(output_dir_path)

		command = f'python3 offlineALL.py --SOUNDFILE_FILEPATH {soundfile_filepath} --CONFIG_FILEPAHT {config_filepath} --OUTPUT_DIR_PATH {output_dir_path}'
		command_strings.append(command)

		#looper = AutonomousLooperOffline(soundfile_filepath, config_filepath=config_filepath, plotFlag=False)
		#looper.computeLooperTrack(output_dir_path)

	# GENERATE LOOPER RESULTS WITH MULTI THREADS
	THREADS = 6
	subdiv = THREADS # num threads
	for i in range(int(len(command_strings) / subdiv)):
		processes = [Popen(command_strings[i*subdiv + j], shell=True) for j in range(subdiv)]
		# collect statuses
		exitcodes = [p.wait() for p in processes]

	# remainder single-exectution
	remaining_indices = len(command_strings) - (i*subdiv+(subdiv-1))
	if remaining_indices > 0:
		for j in range(remaining_indices):
			command = command_strings[(i*subdiv+(subdiv-1)) + j]
			os.system(command)

