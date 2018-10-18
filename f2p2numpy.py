import numpy as np
import argparse
import os

def read_array_from_faust2plot_outfile(file):
	read = False
	array = []
	with open(file) as r:
		for line in r:
			line = line.strip()
			if "faustout" in line:
				read = True
				continue
			if read == True:
				if "];" in line:
					return np.array(array)
				if "Chunk Boundary" in line:
					continue
				string_value = line.split(";")[0]
				float_value = float(string_value)
				array.append(float_value)

def  main(args):
	array = read_array_from_faust2plot_outfile(args.infile)
	np.save(args.outfile, array)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('infile')
	parser.add_argument('--outfile', default='sample')
	args = parser.parse_args()
	main(args)