import os
import json
import numpy as np
from collections import Counter


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

	# open JSON logfile to use for objective
	path_to_objective_log = './corpus/decisions_log.json'
	with open(path_to_objective_log, 'r') as file:
	    objective_log = json.load(file)
	objective_binary_log = decisionLogToBinary(objective_log)

	# open JSON logfiles of generated for search
	looper_outputs_dir = './looper_outputs'
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
		with open(f'./best_configs/{i[0]}.json', 'w', encoding='utf-8') as f:
			json.dump(best_config, f, ensure_ascii=False, indent=4)




