import json
import os
import itertools
import shutil
import sys
import argparse

import numpy as np
from threading import Thread
import subprocess
from subprocess import Popen # process on multiple threads
from collections import Counter

import fitness_functions as fit
import mutations as mut
from constants import RULE_NAMES, XI_VALUES, THRESHOLD_VALUES
from offlineALLclass import AutonomousLooperOffline


def call_script(args):
	#print(args.split(' '))
	subprocess.call(args.split(' '))
	#os.system(args)


if __name__ == '__main__': 

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--ITERATIONS', type=int, default=2,
						help='number of iterations')
	parser.add_argument('--THREADS', type=int, default=6,
						help='number of threads for parallel computation')
	parser.add_argument('--NUM_PARENTS', type=int, default=2,
						help='number of parents selected at each iteration')
	parser.add_argument('--NUM_OFFSPRING', type=int, default=4,
						help='number of offspring generated at each iteration')
	parser.add_argument('--NUM_RANDOM', type=int, default=4,
						help='number of random offspring generated at each iteration')
	args = parser.parse_args(sys.argv[1:])


	## DEFINE SCRIPT PARAMETERS
	N_ITERATIONS = args.ITERATIONS

	NUM_BEST = args.NUM_PARENTS # number of best configs to keep
	#NUM_BEST = 2 # number of best configs to keep
	MULT_FACTOR = args.NUM_OFFSPRING # number of mutated copies for each of best configurations
	#MULT_FACTOR = 4 # number of mutated copies for each of best configurations
	NUM_RANDOM = args.NUM_RANDOM # number of new random elements at each generation
	#NUM_RANDOM = 5 # number of new random elements at each generation
	N_POPULATION = NUM_BEST * (MULT_FACTOR + 1) + NUM_RANDOM

	#THREADS = 6 # for multi-thread computing
	THREADS = args.THREADS # for multi-thread computing


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



	print(f'Generating population of {N_POPULATION} random config files...')
	# INITIALIZE RANDOM CONFIG FILES
	for i in range(N_POPULATION):

		# MAKE BASIC CONFIG FILE WITH RANDOM RULES
		new_rules = []
		rule = mut.makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)
		new_rules.append([mut.makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])
		new_rules.append([mut.makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])

		# generate config files for all rule combinations
		config_file_blank = basic_config_file.copy()
		config_file_blank['looping-rules'] = new_rules
		with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
			json.dump(config_file_blank, f, ensure_ascii=False, indent=4)



	# RUN GENETIC ALGORITHM
	#N_ITERATIONS = 2
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
		threads = []
		for command in command_strings:
			threads.append(Thread(target=call_script, args=([command])))
		# Start all threads
		for w, t in enumerate(threads):
			print(f'Computing ALL with config file {w}...')
			t.start()
		# Wait for all threads to finish
		for t in threads:
			t.join()


		'''
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
		'''




		# EVALUATE FITNESS FUNCTION
		# open JSON logfile to use for objective
		print('Evaluating fitness function...')
		path_to_objective_log = './genetic_algorithm/corpus/decisions_log.json'
		with open(path_to_objective_log, 'r') as file:
		    objective_log = json.load(file)
		#objective_binary_log = decisionLogToBinary(objective_log)

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
			#binary_decisions = decisionLogToBinary(decisions_log)
			# compute fitness function as comparison
			#scores[path] = fit.wightedLoopNumberFitnessFunction(decisions_log, objective_log)
			scores[path] = fit.wightedBinaryFitnessFunction(decisions_log, objective_log, weight=1)



		# SELECT FITTEST
		# Find highest fitness values
		k = Counter(scores)
		high = k.most_common(NUM_BEST)
		print()
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
		i = 0 # count new config files
		best_configs_list = []
		for config_filepath in best_configs_paths:
			# open best JSON config file
			with open(f'{best_configs_path}/{config_filepath}', 'r') as file:
				config_file = json.load(file)
			# keep the best ones in the population
			with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
				json.dump(config_file, f, ensure_ascii=False, indent=4)
			i += 1
			best_configs_list.append(config_file)

		# generate random mutations
		for zz, config_file in enumerate(best_configs_list):
			for _ in range(0,MULT_FACTOR,2):
				# rules of this file
				rules_1 = config_file["looping-rules"]

				# extract random config file from best ones to cross-mutate with
				other_idxs = list(range(len(best_configs_list)))
				other_idxs.remove(zz) # remove self
				config_2 = best_configs_list[zz]
				rules_2 = config_2["looping-rules"]

				new_rules_1, new_rules_2 = mut.crossCombine(rules_1, rules_2)

				new_rules_1 = mut.RandomMutate(new_rules_1)
				new_rules_2 = mut.RandomMutate(new_rules_2)
				config_file["looping-rules"] = new_rules_1
				config_2["looping-rules"] = new_rules_2
				with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
					json.dump(config_file, f, ensure_ascii=False, indent=4)
				i += 1
				with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
					json.dump(config_2, f, ensure_ascii=False, indent=4)
				i += 1

		for _ in range(NUM_RANDOM):
			# MAKE BASIC CONFIG FILE WITH RANDOM RULES
			new_rules = []
			rule = mut.makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)
			new_rules.append([mut.makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])
			new_rules.append([mut.makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])

			# generate config files for all rule combinations
			config_file_blank = basic_config_file.copy()
			config_file_blank['looping-rules'] = new_rules
			with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
				json.dump(config_file_blank, f, ensure_ascii=False, indent=4)
			i += 1





