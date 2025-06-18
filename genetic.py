import json
import os
import itertools
import shutil
import numpy as np
from offlineALLclass import AutonomousLooperOffline
from subprocess import Popen # process on multiple threads
from collections import Counter
import sys
import argparse


# utils
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


def FitnessFunction(binary_decisions, objective_binary_log):
	# compute fitness function
	score = np.array(binary_decisions) + np.array(objective_binary_log)
	unique, counts = np.unique(score, return_counts=True)
	counter = dict(zip(unique, counts))
	try:
		fitness = counter[2]
	except: 
		fitness = 0
	return fitness


# mutation function
def MutationAddRandomRule(rules):
	# add a random rule to a random loop track
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	if len(rules[idx_rule_to_change]) < N_MAX_RULES:
		rules[idx_rule_to_change].append(makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES))
	return rules

def MutationRemoveRandomRule(rules):
	# remove a random rule from a random loop track
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	if len(rules[idx_rule_to_change]) > N_MIN_RULES:
		rules[idx_rule_to_change].remove(rules[idx_rule_to_change][idx_rule_component_to_change])
	return rules

def MutationSubstituteRandomRule(rules):
	# substitute a rule with another random rule
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	rules[idx_rule_to_change][idx_rule_component_to_change] = makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)
	return rules

def MutationIncreaseThreshold(rules):
	# increase the value of a threshold of a random rule element
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	if rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] > 0 and rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] < 1:
		rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] += 0.1
	return rules

def MutationDecreaseThreshold(rules):
	# increase the value of a threshold of a random rule element
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	if rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] > 0 and rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] < 1:
		rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] -= 0.1
	return rules

def RandomMutate(rules, n_mutations=1):
	new_rules = rules.copy()
	for _ in range(n_mutations):
		mutation_type = np.random.randint(0, high=N_MUTATION_TYPES)
		if mutation_type == 0:
			new_rules = MutationAddRandomRule(rules)
		elif mutation_type == 1:
			new_rules = MutationRemoveRandomRule(rules)
		elif mutation_type == 2:
			new_rules = MutationSubstituteRandomRule(rules)
		elif mutation_type == 3:
			new_rules = MutationIncreaseThreshold(rules)
		elif mutation_type == 4:
			new_rules = MutationDecreaseThreshold(rules)
	return new_rules



if __name__ == '__main__': 

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--ITERATIONS', type=int, default=2,
						help='number of iterations')
	parser.add_argument('--THREADS', type=int, default=6,
						help='number of threads for parallel computation')
	args = parser.parse_args(sys.argv[1:])

	## DEFINE SCRIPT PARAMETERS
	iterations = args.ITERATIONS
	threads = args.THREADS


	# INITIALIZE BASIC CONFIG FILE
	# define soundfile and basic config
	soundfile_filepath = './genetic_algorithm/corpus/USE_CASE_1.wav'
	starting_config_filepath = './genetic_algorithm/corpus/objective_config.json'
	
	# open basic JSON config file
	with open(starting_config_filepath, 'r') as file:
	    basic_config_file = json.load(file)
	#print(basic_config_file)
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



	# GENERAL INITS
	config_files_path = './genetic_algorithm/config_files'
	best_configs_path = './genetic_algorithm/best_configs'
	looper_outputs_path = './genetic_algorithm/looper_outputs'
	for f in os.listdir(config_files_path):
		os.remove(os.path.join(config_files_path, f))
	for f in os.listdir(best_configs_path):
		os.remove(os.path.join(best_configs_path, f))
	shutil.rmtree(looper_outputs_path)
	os.mkdir(looper_outputs_path)

	NUM_BEST = 5 # number of best configs to keep
	MULT_FACTOR = 2 # number of mutated copies for each of best configurations
	NUM_RANDOM = 3 # number of new random elements at each generation
	N_POPULATION = NUM_BEST * (MULT_FACTOR + 1) + NUM_RANDOM

	#THREADS = 6 # for multi-thread computing
	THREADS = threads # for multi-thread computing
	N_MAX_RULES = 5 # for computation of mutations
	N_MIN_RULES = 1 # for computation of mutations
	N_MUTATION_TYPES = 5 # for computation of mutations


	print(f'Generating population of {N_POPULATION} random config files...')
	# INITIALIZE RANDOM CONFIG FILES
	for i in range(N_POPULATION):

		# MAKE BASIC CONFIG FILE WITH RANDOM RULES
		new_rules = []
		rule = makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)
		new_rules.append([makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])
		new_rules.append([makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])

		# generate config files for all rule combinations
		config_file_blank = basic_config_file.copy()
		config_file_blank['looping-rules'] = new_rules
		with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
			json.dump(config_file_blank, f, ensure_ascii=False, indent=4)



	# RUN GENETIC ALGORITHM
	#N_ITERATIONS = 2
	N_ITERATIONS = iterations
	for k in range(N_ITERATIONS):

		print(f'Genetic algorithm iteration {k}')
		print('-'*50)

		# GENERATE ALL RESULTS FROM CONFIG FILE
		shutil.rmtree(looper_outputs_path) # remove old files
		os.mkdir(looper_outputs_path) 
		config_files_list = os.listdir(config_files_path) 
		command_strings = []
		# generate commands to run looper for each config file
		for config_file in config_files_list:
			config_filepath = f'{config_files_path}/{config_file}'
			output_dir_path = f'{looper_outputs_path}/{config_file.split(".")[0]}'
			if os.path.isdir(output_dir_path):
				shutil.rmtree(output_dir_path)
			os.mkdir(output_dir_path)

			command = f'python3 offlineALL.py --SOUNDFILE_FILEPATH {soundfile_filepath} --CONFIG_FILEPAHT {config_filepath} --OUTPUT_DIR_PATH {output_dir_path} --VERBOSE 0 --IGNORE_WARNINGS True'
			command_strings.append(command)

		# GENERATE LOOPER RESULTS WITH MULTI THREADS
		subdiv = THREADS # num threads
		for i in range(int(len(command_strings) / subdiv)):
			for j in range(subdiv):
				print(f'Computing ALL with config file {i*subdiv + j}...')
			processes = [Popen(command_strings[i*subdiv + j], shell=True) for j in range(subdiv)]
			# collect statuses
			exitcodes = [p.wait() for p in processes]

		# remainder single-exectution
		remaining_indices = len(command_strings) - (i*subdiv+(subdiv-1))
		if remaining_indices > 0:
			for j in range(remaining_indices):
				print(f'Computing ALL with config file {(i*subdiv+(subdiv-1)) + j}...')
				command = command_strings[(i*subdiv+(subdiv-1)) + j]
				os.system(command)




		# EVALUATE FITNESS FUNCTION
		# open JSON logfile to use for objective
		print('Evaluating fitness function...')
		path_to_objective_log = './genetic_algorithm/corpus/decisions_log.json'
		with open(path_to_objective_log, 'r') as file:
		    objective_log = json.load(file)
		objective_binary_log = decisionLogToBinary(objective_log)

		# open JSON logfiles of generated for search
		for f in os.listdir(best_configs_path):
			os.remove(os.path.join(best_configs_path, f)) # remove previous best configs
		looper_outputs_paths = os.listdir(looper_outputs_path)
		looper_outputs_paths = [path for path in looper_outputs_paths if path != '.DS_Store']
		scores = {}
		for path in looper_outputs_paths:
			# open logifle JSON
			logifle_path = f'{looper_outputs_path}/{path}/USE_CASE_1/decisions_log.json'
			with open(logifle_path, 'r') as file:
			    decisions_log = json.load(file)
			# transform logfile to binary
			binary_decisions = decisionLogToBinary(decisions_log)
			# compute fitness function as comparison
			scores[path] = FitnessFunction(binary_decisions, objective_binary_log)



		# SELECT FITTEST
		# Find highest fitness values
		k = Counter(scores)
		high = k.most_common(NUM_BEST)
		print(f"Configurations with {NUM_BEST} highest scores:")
		for i in high:
			print(f'{i[0]}: {i[1]}')

			# save highest in best_configs folder
			best_config_path = f'{looper_outputs_path}/{i[0]}/USE_CASE_1/config.json'
			with open(best_config_path, 'r') as file:
				best_config = json.load(file)
			with open(f'{best_configs_path}/{i[0]}.json', 'w', encoding='utf-8') as f:
				json.dump(best_config, f, ensure_ascii=False, indent=4)


		print('Computing mutations...')
		print()
		print()
		# COMPUTE MUTATIONS
		for f in os.listdir(config_files_path):
			os.remove(os.path.join(config_files_path, f)) # remove old config files
		best_configs_paths = os.listdir(best_configs_path) 
		i = 0 # count config files
		for config_filepath in best_configs_paths:
			# open best JSON config file
			with open(f'{best_configs_path}/{config_filepath}', 'r') as file:
			    config_file = json.load(file)
			# keep the best ones in the population
			with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
				json.dump(config_file, f, ensure_ascii=False, indent=4)
			i += 1

			# generate random mutations
			for _ in range(MULT_FACTOR):
				rules = config_file["looping-rules"]
				new_rules = RandomMutate(rules)
				config_file["looping-rules"] = new_rules
				with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
					json.dump(config_file, f, ensure_ascii=False, indent=4)
				i += 1

		for _ in range(NUM_RANDOM):
			# MAKE BASIC CONFIG FILE WITH RANDOM RULES
			new_rules = []
			rule = makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)
			new_rules.append([makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])
			new_rules.append([makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])

			# generate config files for all rule combinations
			config_file_blank = basic_config_file.copy()
			config_file_blank['looping-rules'] = new_rules
			with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
				json.dump(config_file_blank, f, ensure_ascii=False, indent=4)
			i += 1





