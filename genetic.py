import json
import os
import itertools
import shutil
import sys
import argparse
import random

import numpy as np
from threading import Thread
import subprocess
from subprocess import Popen # process on multiple threads
from collections import Counter

import fitness_functions as fit
import mutations as mut
from constants import RULE_NAMES, XI_VALUES, THRESHOLD_VALUES
from offlineALLclass import AutonomousLooperOffline


# utils
def call_script(args):
	subprocess.call(args.split(' '))

def fitnessToScore(fitness):
	return np.exp(10*fitness)/100


if __name__ == '__main__': 

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--ITERATIONS', type=int, default=2,
						help='number of iterations')
	parser.add_argument('--NUM_PARENTS', type=int, default=8,
						help='number of parents selected at each iteration')
	parser.add_argument('--NUM_OFFSPRING', type=int, default=10,
						help='number of offspring generated at each iteration')
	parser.add_argument('--NUM_RANDOM', type=int, default=4,
						help='number of random offspring generated at each iteration')
	parser.add_argument('--NUM_MUTATIONS', type=int, default=4,
						help='number of mutations for each new element at each iteration')
	parser.add_argument('--NUM_KEEP', type=int, default=2,
						help='number of best configurations kept at each iteration')
	parser.add_argument('--FITNESS_FUNCTION', type=str, default="binary",
						help='options: binary, weightedBinary, loopNumber, weightedLoopNumber')
	parser.add_argument('--FINTESS_WEIGTH', type=float, default=0.8,
						help='weight to use for weighted fitness functions')
	args = parser.parse_args(sys.argv[1:])


	## DEFINE SCRIPT PARAMETERS
	N_ITERATIONS = args.ITERATIONS
	NUM_BEST = args.NUM_PARENTS # number of parents
	NUM_OFFSPRING = args.NUM_OFFSPRING # number of offspring
	NUM_RANDOM = args.NUM_RANDOM # number of new random generations at each generation	
	NUM_MUTATIONS = args.NUM_MUTATIONS # number of mutations
	FITNESS_FUNCTION = args.FITNESS_FUNCTION # fitness function
	FINTESS_WEIGTH = args.FINTESS_WEIGTH # certain fitness functions require weights
	NUM_KEEP = 2 # number of best configurations to keep at each iteration
	N_POPULATION = NUM_KEEP + NUM_OFFSPRING + NUM_RANDOM


	print()
	print('GENETIC ALGORITHM FOR LEARNING MAAL RULE SETS')
	print('-' * 50)
	print()
	print(f'Computing MAAL output from user corpus...')
	# more than one corpus tracks and generate log before starting 
	# sound examples should all have the same BPM
	sound_corpus_filepath = './genetic_algorithm/corpus/sound' # dir where all the sound examples are
	starting_config_filepath = './genetic_algorithm/corpus/objective_config.json'
	corpus_output_dir_path = './genetic_algorithm/corpus/looper_outputs'
	threads = []
	for soundfile in os.listdir(sound_corpus_filepath):
		if soundfile != '.DS_Store':
			print(f'Computing MAAL with sound: {soundfile}')
			command = f'python3 offlineALL.py --SOUNDFILE_FILEPATH {sound_corpus_filepath}/{soundfile} --CONFIG_FILEPAHT {starting_config_filepath} --OUTPUT_DIR_PATH {corpus_output_dir_path} --VERBOSE 0 --IGNORE_WARNINGS True'
			threads.append(Thread(target=call_script, args=([command])))
	[t.start() for t in threads]
	[t.join() for t in threads]
	print()


	# INITIALIZE BASIC CONFIG FILE
	# define soundfile and basic config
	# open basic JSON config file
	with open(starting_config_filepath, 'r') as file:
	    basic_config_file = json.load(file)
	rules = basic_config_file['looping-rules']


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
	if os.path.isdir(config_files_path):
		shutil.rmtree(config_files_path)
		os.mkdir(config_files_path)
	else:
		os.mkdir(config_files_path)
	if os.path.isdir(best_configs_path):
		shutil.rmtree(best_configs_path)
		os.mkdir(best_configs_path)
	else:
		os.mkdir(best_configs_path)
	if os.path.isdir(looper_outputs_path):
		shutil.rmtree(looper_outputs_path)
		os.mkdir(looper_outputs_path)
	else:
		os.mkdir(looper_outputs_path)

	# INITIALIZE RANDOM CONFIG FILES
	print(f'Generating population of {N_POPULATION} random config files...')
	for i in range(N_POPULATION):

		# make basic config file with random rules
		new_rules = []
		for _ in range(len(rules)):
			new_rules.append([mut.makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)])

		# generate config files for all rule combinations
		config_file_blank = basic_config_file.copy()
		config_file_blank['looping-rules'] = new_rules
		with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
			json.dump(config_file_blank, f, ensure_ascii=False, indent=4)


	# RUN GENETIC ALGORITHM
	soundfile_filepath = './genetic_algorithm/corpus/USE_CASE_2.wav'
	for k in range(N_ITERATIONS):

		print(f'Genetic algorithm iteration {k}')
		print('-'*50)

		# GENERATE ALL RESULTS FROM NEW POPULATION OF CONFIG FILES
		shutil.rmtree(looper_outputs_path) # remove old files
		os.mkdir(looper_outputs_path) 
		config_files_list = os.listdir(config_files_path) 
		threads = []
		# generate commands to run looper for each config file
		for config_file in config_files_list:
			config_filepath = f'{config_files_path}/{config_file}'
			output_dir_path = f'{looper_outputs_path}/{config_file.split(".")[0]}'
			if os.path.isdir(output_dir_path):
				shutil.rmtree(output_dir_path)
			os.mkdir(output_dir_path)
			print(f'Computing MAAL with {config_file}')
			for soundfile in os.listdir(sound_corpus_filepath):
				if soundfile != '.DS_Store':
					command = f'python3 offlineALL.py --SOUNDFILE_FILEPATH {sound_corpus_filepath}/{soundfile} --CONFIG_FILEPAHT {config_filepath} --OUTPUT_DIR_PATH {output_dir_path} --VERBOSE 0 --IGNORE_WARNINGS True'
					threads.append(Thread(target=call_script, args=([command])))
		[t.start() for t in threads]
		[t.join() for t in threads]


		# EVALUATE FITNESS FUNCTION
		# open JSON logfile to use for objective
		print('Evaluating fitness function...')

		# open JSON logfiles of generated for search
		for f in os.listdir(best_configs_path):
			os.remove(os.path.join(best_configs_path, f)) # remove previous best configs
		looper_outputs_paths = os.listdir(looper_outputs_path)
		looper_outputs_paths = [path for path in looper_outputs_paths if path != '.DS_Store']

		scores = {}
		for path in looper_outputs_paths:
			config_score = 0
			for soundfile_name in os.listdir(f'{looper_outputs_path}/{path}'):

				# open objective logifle JSON
				objective_logifle_path = f'{corpus_output_dir_path}/{soundfile_name}/decisions_log.json'
				with open(objective_logifle_path, 'r') as file:
				    objective_log = json.load(file)

				# open generated logifle JSON
				decisions_logifle_path = f'{looper_outputs_path}/{path}/{soundfile_name}/decisions_log.json'
				with open(decisions_logifle_path, 'r') as file:
				    decisions_log = json.load(file)

				if FITNESS_FUNCTION == "binary":
					config_score += fit.binaryFitnessFunction(decisions_log, objective_log)
				elif FITNESS_FUNCTION == "weightedBinary":
					config_score += fit.weightedBinaryFitnessFunction(decisions_log, objective_log, weight=FINTESS_WEIGTH)
				elif FITNESS_FUNCTION == "loopNumber":
					config_score += fit.loopNumberFitnessFunction(decisions_log, objective_log)
				elif FITNESS_FUNCTION == "weightedLoopNumber":
					config_score += fit.weightedLoopNumberFitnessFunction(decisions_log, objective_log, weight=FINTESS_WEIGTH)

			scores[path] = config_score / len(os.listdir(f'{looper_outputs_path}/{path}')) # normalize score by number of soundfiles in corpus


		# SELECTION FITTEST
		# Select highest fitness values
		k = Counter(scores)
		high = k.most_common(NUM_BEST)
		print()
		print(f"Configurations with {NUM_BEST} highest scores:")
		config_names = []
		config_scores = []
		for i in high:
			print(f'{i[0]}: {i[1]:.3f} - score {fitnessToScore(i[1]):.2f}')
			config_names.append(i[0])
			config_scores.append(fitnessToScore(i[1]))

			# transfer fittest in best_configs folder
			best_config_path = f'{config_files_path}/{i[0]}.json'
			with open(best_config_path, 'r') as file:
				best_config = json.load(file)
			with open(f'{best_configs_path}/{i[0]}.json', 'w', encoding='utf-8') as f:
				json.dump(best_config, f, ensure_ascii=False, indent=4)


		# keep the best NUM_KEEP config files
		i = 0 # count new config files
		for ii in range(NUM_KEEP):
			# open best JSON config file
			with open(f'{best_configs_path}/{config_names[ii]}.json', 'r') as file:
				config_file = json.load(file)
			# keep the best ones in the population
			with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
				json.dump(config_file, f, ensure_ascii=False, indent=4)
			i += 1

		# compute mutations
		print('Computing mutations...')
		print()
		print()
		for _ in range(0, NUM_OFFSPRING, 2):
			# extract two random parents according to the weights probabilities
			configs = random.choices(config_names, weights=config_scores, k=2)

			# open config files
			with open(f'{best_configs_path}/{configs[0]}.json', 'r') as file:
				config_1 = json.load(file)
			with open(f'{best_configs_path}/{configs[1]}.json', 'r') as file:
				config_2 = json.load(file)

			# extract rules
			rules_1 = config_1["looping-rules"]
			rules_2 = config_2["looping-rules"]

			# crossover
			new_rules_1, new_rules_2 = mut.crossCombine(rules_1, rules_2)
			# mutation
			new_rules_1 = mut.RandomMutate(new_rules_1, n_mutations=NUM_MUTATIONS)
			new_rules_2 = mut.RandomMutate(new_rules_2, n_mutations=NUM_MUTATIONS)

			config_file["looping-rules"] = new_rules_1
			config_2["looping-rules"] = new_rules_2
			with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
				json.dump(config_file, f, ensure_ascii=False, indent=4)
			i += 1
			with open(f'{config_files_path}/config_{i}.json', 'w', encoding='utf-8') as f:
				json.dump(config_2, f, ensure_ascii=False, indent=4)
			i += 1


		# MAKE SOME NEW CONFIG FILES WITH COMPLETELY RANDOM RULES
		for _ in range(NUM_RANDOM):
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



