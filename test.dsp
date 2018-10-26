
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
dx.dx7_algo(1,egR1,egR2,egR3,egR4,egL1,egL2,egL3,egL4,outLevel,keyVelSens,ampModSens,opMode,opFreq,opDetune,opRateScale,feedback,lfoDelay,lfoDepth,lfoSpeed,freq,gain,gate)
with{
	egR1(n) = ba.take(n+1,(80*2,53,54,56,76,99));
	egR2(n) = ba.take(n+1,(56,46,15,74,73,76));
	egR3(n) = ba.take(n+1,(10,32,10,10,10,10));
	egR4(n) = ba.take(n+1,(45,61,47,45,55,32));
	egL1(n) = ba.take(n+1,(98,99,99,98,99,99));
	egL2(n) = ba.take(n+1,(98,93,92,98,92,92));
	egL3(n) = ba.take(n+1,(36,90,0,36,0,0));
	egL4(n) = ba.take(n+1,(0,0,0,0,0,0));
	outLevel(n) = ba.take(n+1,(99,83,96,72,80,82));
	// keyVelSens(n) = ba.take(n+1,(0,0,0,0,0,0));
	keyVelSens(n) = ba.take(n+1,(2,2,3,1,1,1));
	// keyVelSens(n) = ba.take(n+1,(6,6,8,4,4,4)); // zero to 8
	ampModSens(n) = ba.take(n+1,(0,0,0,0,0,0));
	opMode(n) = ba.take(n+1,(0,0,0,0,0,0));
	opFreq(n) = ba.take(n+1,(1,1,2,2,2,2));
	opDetune(n) = ba.take(n+1,(0,-6,1,0,0,0));
	opRateScale(n) = ba.take(n+1,(0,0,0,0,0,0));
	// feedback = 7 : dx.dx7_fdbkscalef/(2*ma.PI);
	lfoDelay = 63;
	lfoDepth = 6;
	lfoSpeed = 30;
	feedback = 10;
	// lfoDepth = hslider("lfoDepth[OWL:B]",0,0,99,1);
	// lfoSpeed = hslider("lfoSpeed[OWL:C]",0,0,99,1);
};

dx7patch = dx7_ORCHESTRA(freq,gain,gate)
with {
     freq = 130;
     gain = .5;
     gate = 1;
};

process = dx7patch;