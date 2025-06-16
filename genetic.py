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
	THRESHOLD_VALUES = np.arange(0.0, 1.0+step, step)

	print("Rule names:")
	print(RULE_NAMES)
	print("Xi values:")
	print(XI_VALUES)
	print("Threshold values:")
	print(THRESHOLD_VALUES)

	def makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES):
		rule = {}
		rule["rule-name"] = RULE_NAMES[np.random.randint(0, high=len(RULE_NAMES))]
		rule["rule-type"] = XI_VALUES[np.random.randint(0, high=len(XI_VALUES))]
		rule["rule-threshold"] = THRESHOLD_VALUES[np.random.randint(0, high=len(THRESHOLD_VALUES))]
		return rule

	new_rules = []
	rule = makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)
	new_rules.append(makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES))
	new_rules.append(makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES))
	print(rule)

	# generate config files for all rule combinations
	config_files_path = './genetic_algorithm/config_files'
	config_file_blank = basic_config_file.copy()
	config_file_blank['looping-rules'] = new_rules
	with open(f'{config_files_path}/config_new.json', 'w', encoding='utf-8') as f:
		json.dump(config_file_blank, f, ensure_ascii=False, indent=4)


	# mutations: add a new random rule, change threshold (higher or lower), change less to more (does it make sense?)
	# probability of mutations




