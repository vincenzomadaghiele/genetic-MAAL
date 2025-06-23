import json
import numpy as np



# process decisions log
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

def decisionLogToTrackNum(decisions_log):
	# convert to binary array
	binary_decisions = []
	for decision in decisions_log:
		outcome = False
		for d in decision['decisions']:
			if d['decision_type'] == 'I' or d['decision_type'] == 'A':
				outcome = True
		if outcome:
			binary_decisions.append(d["loop_track (i)"] + 1)
		else:
			binary_decisions.append(0)
	return binary_decisions


# fitness functions
def binaryFitnessFunction(comparable_log, objective_log):
	comparable_binary = decisionLogToBinary(comparable_log)
	objective_binary = decisionLogToBinary(objective_log)
	return np.equal(np.array(comparable_binary), np.array(objective_binary)).sum() / np.shape(objective_binary)[0]


def wightedBinaryFitnessFunction(comparable_log, objective_log, weight=0.5):
	# the weight makes 0s in the score count less
	comparable_binary = np.array(decisionLogToBinary(comparable_log))
	objective_binary = np.array(decisionLogToBinary(objective_log))
	score = 0
	for i in range((objective_binary).shape[0]):
		if comparable_binary[i] == objective_binary[i]:
			if objective_binary[i] != 0:
				score += 1
			else:
				score += weight
	objective_binary = np.where(objective_binary==0, weight, objective_binary)
	max_score = objective_binary.sum()
	return score / max_score


def loopNumberFitnessFunction(comparable_log, objective_log):
	comparable_binary = decisionLogToTrackNum(comparable_log)
	objective_binary = decisionLogToTrackNum(objective_log)
	return np.equal(np.array(comparable_binary), np.array(objective_binary)).sum() / np.shape(objective_binary)[0]


def wightedLoopNumberFitnessFunction(comparable_log, objective_log, weight=0.5):
	# the weight makes 0s in the score count less
	comparable_binary = np.array(decisionLogToTrackNum(comparable_log))
	objective_binary = np.array(decisionLogToTrackNum(objective_log))
	score = 0
	for i in range((objective_binary).shape[0]):
		if comparable_binary[i] == objective_binary[i]:
			if objective_binary[i] != 0:
				score += 1
			else:
				score += weight
	objective_binary_count = np.array(decisionLogToBinary(objective_log))
	objective_binary_count = np.where(objective_binary_count==0, weight, objective_binary_count)
	max_score = objective_binary_count.sum()
	return score / max_score




if __name__ == '__main__': 

	# load comparable log
	logifle_path = f'genetic_algorithm/looper_outputs/config_0/USE_CASE_1/decisions_log.json'
	with open(logifle_path, 'r') as file:
		comparable_log = json.load(file)
	# load comparable log
	logifle_path = f'genetic_algorithm/corpus/decisions_log.json'
	with open(logifle_path, 'r') as file:
		objective_log = json.load(file)	

	# compute fitness function as comparison
	print(f'Binary fitness functon: {binaryFitnessFunction(comparable_log, objective_log):.3f}')
	print(f'Loop track number fitness functon: {loopNumberFitnessFunction(comparable_log, objective_log):.3f}')
	print(f'Weighted binary fitness functon: {wightedBinaryFitnessFunction(comparable_log, objective_log):.3f}')
	print(f'Weighted loop track number fitness functon: {wightedLoopNumberFitnessFunction(comparable_log, objective_log):.3f}')





