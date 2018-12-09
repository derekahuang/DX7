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
dx.dx7_algo(%d,egR1,egR2,egR3,egR4,egL1,egL2,egL3,egL4,outLevel,keyVelSens,ampModSens,opMode,opFreq,opDetune,opRateScale,feedback,lfoDelay,lfoDepth,lfoSpeed,freq,gain,gate)
with{
	egR1(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egR2(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egR3(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egR4(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egL1(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egL2(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egL3(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	egL4(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	outLevel(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	keyVelSens(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	ampModSens(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	opMode(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	opFreq(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	opDetune(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	opRateScale(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	feedback(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	lfoDelay(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	lfoDepth(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
	lfoSpeed(n) = ba.take(n+1,(%d,%d,%d,%d,%d,%d));
};

dx7patch = dx7_ORCHESTRA(freq,gain,gate)
with {
     freq = %d;
     gain = .5;
     gate = 1;
};

process = dx7patch;"""

params = {
	"egR1" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"egR2" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"egR3" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"egR4" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"egL1" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"egL2" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"egL3" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"egL4" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"outLevel" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"keyVelSens" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"ampModSens" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"opMode" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 2},
	"opFreq" : {"dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	"opDetune" : {"dist" : np.random.randint, "size": (6), "low": 1, "high": 100},
	"opRateScale" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"feedback" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"lfoDelay" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"lfoDepth" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"lfoSpeed" : {"dist" : np.random.randint, "size": (6), "low": 0, "high": 100},
	"freq" : {"dist" : np.random.randint, "size": (1), "low": 1, "high": 88},
}


def equation(x):
	return 440 * (2 ** (1/float(12))) ** (x - 49)

def main(args):
	generated = []
	count = 0
	while count < args.num_samples:
		if args.alg is None:
			param_list = np.random.randint(0, high=32, size=(1))
		else:
			param_list = [args.alg]
		param_dict = {"alg": args.alg}
		for k, v in params.items():
			sample = v["dist"](v["low"], high=v["high"], size=v["size"])
			if k == "freq":
				sample = equation(sample)
			sample = list(sample)
			param_list += sample
			param_dict[k] = sample
		temp = dx % tuple(param_list)
		if temp in generated:
			continue
		generated.append()
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
	args = parser.parse_args()
	main(args)