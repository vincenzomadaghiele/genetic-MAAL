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


if __name__ == '__main__': 

	path_to_objective_log = './01_output_offline/USE CASE 1/decisions_log.json'
	# open logifle JSON
	with open(path_to_objective_log, 'r') as file:
	    objective_log = json.load(file)
	objective_binary_log = decisionLogToBinary(objective_log)
	print(objective_binary_log)


	looper_outputs_dir = './exhaustive_search/looper_outputs'
	looper_outputs_paths = os.listdir(looper_outputs_dir) 
	looper_outputs_paths = [path for path in looper_outputs_paths if path != '.DS_Store']
	print(looper_outputs_paths)

	scores = {}
	for path in looper_outputs_paths:

		# open logifle JSON
		logifle_path = f'{looper_outputs_dir}/{path}/USE_CASE_1/decisions_log.json'
		with open(logifle_path, 'r') as file:
		    decisions_log = json.load(file)

		binary_decisions = decisionLogToBinary(decisions_log)

		# compute similarity score
		score = np.array(binary_decisions) + np.array(objective_binary_log)
		unique, counts = np.unique(score, return_counts=True)
		counter = dict(zip(unique, counts))
		score = counter[2]
		print(f'{path}: {score}')
		scores[path] = score


	k = Counter(scores)

	# Finding 3 highest values
	high = k.most_common(3) 
	print("Dictionary with 3 highest values:")
	print("Keys: Values")
	for i in high:
	    print(i[0]," :",i[1]," ")

