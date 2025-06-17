import json
import os
import itertools
import shutil
import numpy as np
from offlineALLclass import AutonomousLooperOffline
from subprocess import Popen # process on multiple threads
from collections import Counter


def makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES):
	rule = {}
	rule["rule-name"] = RULE_NAMES[np.random.randint(0, high=len(RULE_NAMES))]
	rule["rule-type"] = XI_VALUES[np.random.randint(0, high=len(XI_VALUES))]
	rule["rule-threshold"] = THRESHOLD_VALUES[np.random.randint(0, high=len(THRESHOLD_VALUES))]
	return rule

def decisionLogToBinary(decisions_log):
	# convert to binary array
	binary_decisions = []
	for decision in decisions_log:
		outcome = False
		for d in decision['decisions']:
			if d['decision_type'] == 'I' or d['decision_type'] == 'A':
				outcome = True
		if outcome:
			binary_decisions.append(1)
		else:
			binary_decisions.append(0)
	return binary_decisions

def FitnessFunction(binary_log_1, binary_log_2):
	# compute fitness function
	score = np.array(binary_decisions) + np.array(objective_binary_log)
	unique, counts = np.unique(score, return_counts=True)
	counter = dict(zip(unique, counts))
	return counter[2]



if __name__ == '__main__': 



	# INITIALIZE BASIC CONFIG FILE
	# define soundfile and basic config
	soundfile_filepath = './genetic_algorithm/corpus/USE_CASE_1.wav'
	starting_config_filepath = './genetic_algorithm/corpus/objective_config.json'
	
	# open basic JSON config file
	with open(starting_config_filepath, 'r') as file:
	    basic_config_file = json.load(file)
	print(basic_config_file)
	rules = basic_config_file['looping-rules']
	N_RULES = 0
	for rule in rules:
		for rule_component in rule:
			N_RULES += 1



	# INITIALIZE POSSIBLE SYSTEM SETTINGS
	print()
	RULE_NAMES = [
					"Harmonic similarity", "Harmonic movement - C", "Harmonic movement - D",
					"Melodic similarity", "Melodic trajectory - C", "Melodic trajectory - D",
					"Dynamic similarity", "Dynamic changes - C", "Dynamic changes - D",
					"Timbral similarity", "Timbral evolution - C", "Timbral evolution - D",
					"Global spectral overlap", "Frequency range overlap",
					"Rhythmic similarity", "Rhythmic density",
					"Harmonic function similarity", "Harmonic function transitions - C", "Harmonic function transitions - D"
					]
	XI_VALUES = ["more", "less"]
	step = 0.1
	THRESHOLD_VALUES = np.arange(0.0, 1.0+step, step).tolist()

	print("Rule names:")
	print(RULE_NAMES)
	print("Xi values:")
	print(XI_VALUES)
	print("Threshold values:")
	print(THRESHOLD_VALUES)



	# INITIALIZE CONFIG FILES
	N_POPULATION = 10
	for i in range(N_POPULATION):

		# MAKE BASIC CONFIG FILE WITH RANDOM RULES
		new_rules = []
		rule = makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)
		new_rules.append([makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])
		new_rules.append([makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])
		print(rule)

		# generate config files for all rule combinations
		config_files_path = './genetic_algorithm/config_files'
		config_file_blank = basic_config_file.copy()
		config_file_blank['looping-rules'] = new_rules
		with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
			json.dump(config_file_blank, f, ensure_ascii=False, indent=4)



	# GENERATE RESULTS
	# generate commands to run looper for each config file
	config_files_list = os.listdir(config_files_path) 
	command_strings = []
	for config_file in config_files_list:
		config_filepath = f'{config_files_path}/{config_file}'
		output_dir_path = f'./genetic_algorithm/looper_outputs/{config_file.split(".")[0]}'
		if os.path.isdir(output_dir_path):
			shutil.rmtree(output_dir_path)
		os.mkdir(output_dir_path)

		command = f'python3 offlineALL.py --SOUNDFILE_FILEPATH {soundfile_filepath} --CONFIG_FILEPAHT {config_filepath} --OUTPUT_DIR_PATH {output_dir_path}'
		command_strings.append(command)

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



	# EVALUATE FITNESS FUNCTION
	# open JSON logfile to use for objective
	path_to_objective_log = './genetic_algorithm/corpus/decisions_log.json'
	with open(path_to_objective_log, 'r') as file:
	    objective_log = json.load(file)
	objective_binary_log = decisionLogToBinary(objective_log)

	# open JSON logfiles of generated for search
	looper_outputs_dir = './genetic_algorithm/looper_outputs'
	looper_outputs_paths = os.listdir(looper_outputs_dir) 
	looper_outputs_paths = [path for path in looper_outputs_paths if path != '.DS_Store']
	scores = {}
	for path in looper_outputs_paths:
		# open logifle JSON
		logifle_path = f'{looper_outputs_dir}/{path}/USE_CASE_1/decisions_log.json'
		with open(logifle_path, 'r') as file:
		    decisions_log = json.load(file)
		# transform logfile to binary
		binary_decisions = decisionLogToBinary(decisions_log)
		# compute fitness function as comparison
		scores[path] = FitnessFunction(binary_decisions, objective_binary_log)



	# SELECT FITTEST
	# Find highest fitness values
	NUM_BEST = 5
	k = Counter(scores)
	high = k.most_common(NUM_BEST) 
	print(f"Configurations with {NUM_BEST} highest scores:")
	for i in high:
		print(f'{i[0]}: {i[1]}')

		# save highest in best_configs folder
		best_config_path = f'{looper_outputs_dir}/{i[0]}/USE_CASE_1/config.json'
		with open(best_config_path, 'r') as file:
			best_config = json.load(file)
		with open(f'./genetic_algorithm/best_configs/{i[0]}.json', 'w', encoding='utf-8') as f:
			json.dump(best_config, f, ensure_ascii=False, indent=4)



	# COMPUTE MUTATIONS
	



	# mutations: add a new random rule, change threshold (higher or lower), change less to more (does it make sense?)
	# probability of mutations







