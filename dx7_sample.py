import os
from tqdm import tqdm
import argparse
import numpy as np
import json

dx = """
import("stdfaust.lib");
an = library("analyzers.lib");
ba = library("basics.lib");
co = library("compressors.lib");
de = library("delays.lib");
dm = library("demos.lib");
dx = library("dx7.lib");
en = library("envelopes.lib");
fi = library("filters.lib");
ho = library("hoa.lib");
ma = library("maths.lib");
ef = library("misceffects.lib");
os = library("oscillators.lib");
no = library("noises.lib");
pf = library("phaflangers.lib");
pm = library("physmodels.lib");
re = library("reverbs.lib");
ro = library("routes.lib");
sp = library("spats.lib");
si = library("signals.lib");
sy = library("synths.lib");
ve = library("vaeffects.lib");

dx7_ORCHESTRA(freq,gain,gate) = 
dx.dx7_algo(%i,egR1,egR2,egR3,egR4,egL1,egL2,egL3,egL4,outLevel,keyVelSens,ampModSens,opMode,opFreq,opDetune,opRateScale,feedback,lfoDelay,lfoDepth,lfoSpeed,freq,gain,gate)
with{
	//egR1(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egR1(n) = ba.take(n+1,(80*2,53,54,56,76,99));
	//egR2(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egR2(n) = ba.take(n+1,(56,46,15,74,73,76));
	//egR3(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egR3(n) = ba.take(n+1,(10,32,10,10,10,10));
	//egR4(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egR4(n) = ba.take(n+1,(45,61,47,45,55,32));
	//egL1(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egL1(n) = ba.take(n+1,(98,99,99,98,99,99));
	//egL2(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egL2(n) = ba.take(n+1,(98,93,92,98,92,92));
	//egL3(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egL3(n) = ba.take(n+1,(36,90,0,36,0,0));
	//egL4(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egL4(n) = ba.take(n+1,(1,0,0,0,0,0));
	//outLevel(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	outLevel(n) = ba.take(n+1,(99,83,96,72,80,82));
	//keyVelSens(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	keyVelSens(n) = ba.take(n+1,(2,2,3,1,1,1));
	//ampModSens(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	ampModSens(n) = ba.take(n+1,(0,0,0,0,0,0));
	opMode(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	opFreq(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	opDetune(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	opRateScale(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	feedback = %d;
	lfoDelay = %d;
	lfoDepth = %d;
	lfoSpeed = %d;
};

dx7patch = dx7_ORCHESTRA(freq,gain,gate)
with {
     freq = %d;
     gain = .5;
     gate = 1;
};

process = dx7patch;"""

params = {
	1 : {"name": "egR1", "dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	2 : {"name": "egR2","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	3 : {"name": "egR3","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	4 : {"name": "egR4","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	5 : {"name": "egL1","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	6 : {"name": "egL2","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	7 : {"name": "egL3","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	8 : {"name": "egL4","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	9 : {"name": "outLevel","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	10 : {"name": "keyVelSens","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	11 : {"name": "ampModSens","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	12 : {"name": "opMode","dist" : np.random.randint, "size": (6), "low": 0, "high": 1},
	13 : {"name": "opFreq","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	14 : {"name": "opDetune","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	15 : {"name": "opRateScale","dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	16 : {"name": "feedback","dist" : np.random.randint, "size": (1), "low": 1, "high": 100},
	17 : {"name": "lfoDelay","dist" : np.random.randint, "size": (1), "low": 1, "high": 100},
	18 : {"name": "lfoDepth","dist" : np.random.randint, "size": (1), "low": 1, "high": 100},
	19 : {"name": "lfoSpeed","dist" : np.random.randint, "size": (1), "low": 1, "high": 100},
	20 : {"name": "freq","dist" : np.random.randint, "size": (1), "low": 1, "high": 88},
}


def equation(x):
	return 440 * (2 ** (1/float(12))) ** (x - 49)

def main(args):
	np.random.seed(7)
	generated = []
	count = 0
	while count < args.num_samples:
		if args.alg is None:
			param_list = list(np.random.randint(0, high=32, size=(1)))
		else:
			param_list = [args.alg]
		param_dict = {"alg": args.alg}
		for k, v in sorted(params.items()):
			print(v["name"])
			sample = v["dist"](v["low"], high=v["high"], size=v["size"])
			if v["name"] == "freq":
				sample = equation(sample)
			sample = list(sample)
			try:
				param_list += sample
			except:
				print param_list
				print sample
			param_dict[v["name"]] = sample
		temp = dx % tuple(param_list)

		print(temp)

		if temp in generated:
			continue
		generated.append(temp)
		param_dict_string = json.dumps(param_dict)

		with open("test.dsp", "w") as f:
			f.write(temp)
			f.close()
		os.system('faust -a plot.cpp -o test.cpp test.dsp && '
			'sed -i \'.bak\' \'s/44100/8000/g\' test.cpp && '
			'echo {1} >> {0} && '
			'g++ -Wall -g -lm -lpthread test.cpp -o test && '
			'./test -n 8000 >> {0}'.format(args.outfile, param_dict_string))
		count += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--outfile', type=str, default='output.txt')
	parser.add_argument('--num_samples', type=int, default=30000)
	parser.add_argument('--alg', type=int, default=None)
	args = parser.parse_args()
	main(args)