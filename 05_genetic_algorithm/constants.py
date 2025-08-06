import numpy as np


RULE_NAMES = [
				"Harmonic similarity", "Harmonic movement - C", "Harmonic movement - D",
				"Melodic similarity", "Melodic trajectory - C", "Melodic trajectory - D",
				"Dynamic similarity", "Dynamic changes - C", "Dynamic changes - D",
				"Timbral similarity", "Timbral evolution - C", "Timbral evolution - D",
				"Global spectral overlap", "Frequency range overlap",
				"Rhythmic similarity", "Rhythmic density",
				#"Harmonic function similarity", "Harmonic function transitions - C", "Harmonic function transitions - D"
				]

XI_VALUES = ["more", "less"]

step = 0.1
THRESHOLD_VALUES = np.arange(0.1, 1.0, step).tolist()

N_MAX_RULES = 4 # for computation of mutations
N_MIN_RULES = 1 # for computation of mutations


