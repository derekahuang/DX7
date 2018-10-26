/* ------------------------------------------------------------
name: "test"
Code generated with Faust 2.11.3 (https://faust.grame.fr)
Compilation options: cpp, -scal -ftz 0
------------------------------------------------------------ */

#ifndef  __mydsp_H__
#define  __mydsp_H__

#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif 

#include <algorithm>
#include <cmath>
#include <math.h>


class mydspSIG0 {
	
  private:
	
	int iRec2[2];
	
  public:
	
	int getNumInputsmydspSIG0() {
		return 0;
		
	}
	int getNumOutputsmydspSIG0() {
		return 1;
		
	}
	int getInputRatemydspSIG0(int channel) {
		int rate;
		switch (channel) {
			default: {
				rate = -1;
				break;
			}
			
		}
		return rate;
		
	}
	int getOutputRatemydspSIG0(int channel) {
		int rate;
		switch (channel) {
			case 0: {
				rate = 0;
				break;
			}
			default: {
				rate = -1;
				break;
			}
			
		}
		return rate;
		
	}
	
	void instanceInitmydspSIG0(int samplingFreq) {
		for (int l1 = 0; (l1 < 2); l1 = (l1 + 1)) {
			iRec2[l1] = 0;
			
		}
		
	}
	
	void fillmydspSIG0(int count, float* output) {
		for (int i = 0; (i < count); i = (i + 1)) {
			iRec2[0] = (iRec2[1] + 1);
			output[i] = std::sin((9.58738019e-05f * float((iRec2[0] + -1))));
			iRec2[1] = iRec2[0];
			
		}
		
	}
};

mydspSIG0* newmydspSIG0() { return (mydspSIG0*)new mydspSIG0(); }
void deletemydspSIG0(mydspSIG0* dsp) { delete dsp; }

static float mydsp_faustpower8_f(float value) {
	return (((((((value * value) * value) * value) * value) * value) * value) * value);
	
}
static float ftbl0mydspSIG0[65536];

#ifndef FAUSTCLASS 
#define FAUSTCLASS mydsp
#endif
#ifdef __APPLE__ 
#define exp10f __exp10f
#define exp10 __exp10
#endif

class mydsp : public dsp {
	
 private:
	
	int fSamplingFreq;
	float fConst0;
	int iConst1;
	float fConst2;
	float fRec1[2];
	float fConst3;
	float fConst4;
	float fConst5;
	float fConst6;
	float fConst7;
	float fRec3[2];
	int iConst8;
	float fConst9;
	float fRec6[2];
	float fConst10;
	float fConst11;
	float fConst12;
	float fConst13;
	float fConst14;
	float fConst15;
	float fRec7[2];
	float fRec4[2];
	int iConst16;
	float fConst17;
	float fRec9[2];
	float fConst18;
	float fConst19;
	float fConst20;
	float fConst21;
	float fConst22;
	float fConst23;
	float fRec10[2];
	int iConst24;
	float fConst25;
	float fRec12[2];
	float fConst26;
	float fConst27;
	float fConst28;
	float fConst29;
	float fConst30;
	float fRec13[2];
	float fConst31;
	float fRec15[2];
	float fConst32;
	float fConst33;
	float fConst34;
	float fConst35;
	float fConst36;
	float fConst37;
	float fRec17[2];
	float fConst38;
	float fConst39;
	float fConst40;
	float fConst41;
	
 public:
	
	void metadata(Meta* m) { 
		m->declare("basics.lib/name", "Faust Basic Element Library");
		m->declare("basics.lib/version", "0.0");
		m->declare("envelopes.lib/author", "GRAME");
		m->declare("envelopes.lib/copyright", "GRAME");
		m->declare("envelopes.lib/license", "LGPL with exception");
		m->declare("envelopes.lib/name", "Faust Envelope Library");
		m->declare("envelopes.lib/version", "0.0");
		m->declare("filename", "test");
		m->declare("maths.lib/author", "GRAME");
		m->declare("maths.lib/copyright", "GRAME");
		m->declare("maths.lib/license", "LGPL with exception");
		m->declare("maths.lib/name", "Faust Math Library");
		m->declare("maths.lib/version", "2.1");
		m->declare("name", "test");
		m->declare("oscillators.lib/name", "Faust Oscillator Library");
		m->declare("oscillators.lib/version", "0.0");
	}

	virtual int getNumInputs() {
		return 0;
		
	}
	virtual int getNumOutputs() {
		return 1;
		
	}
	virtual int getInputRate(int channel) {
		int rate;
		switch (channel) {
			default: {
				rate = -1;
				break;
			}
			
		}
		return rate;
		
	}
	virtual int getOutputRate(int channel) {
		int rate;
		switch (channel) {
			case 0: {
				rate = 1;
				break;
			}
			default: {
				rate = -1;
				break;
			}
			
		}
		return rate;
		
	}
	
	static void classInit(int samplingFreq) {
		mydspSIG0* sig0 = newmydspSIG0();
		sig0->instanceInitmydspSIG0(samplingFreq);
		sig0->fillmydspSIG0(65536, ftbl0mydspSIG0);
		deletemydspSIG0(sig0);
		
	}
	
	virtual void instanceConstants(int samplingFreq) {
		fSamplingFreq = samplingFreq;
		fConst0 = std::min<float>(192000.0f, std::max<float>(1.0f, float(fSamplingFreq)));
		iConst1 = (0.0f < (0.576006711f * fConst0));
		fConst2 = (51.5640831f * fConst0);
		fConst3 = (0.00309819565f * fConst0);
		fConst4 = (0.0193945095f / fConst0);
		fConst5 = (0.00209819572f * fConst0);
		fConst6 = (40868.4492f / fConst0);
		fConst7 = (440.0f / fConst0);
		iConst8 = (0.0f < (0.0627472401f * fConst0));
		fConst9 = (0.206408337f * fConst0);
		fConst10 = (0.0630281121f * fConst0);
		fConst11 = (6.97446251f / fConst0);
		fConst12 = (0.0251072664f * fConst0);
		fConst13 = (26.3707199f / fConst0);
		fConst14 = (2892.58887f / fConst0);
		fConst15 = (438.5f / fConst0);
		iConst16 = (0.0f < (0.00100000005f * fConst0));
		fConst17 = (93.754097f * fConst0);
		fConst18 = (2.78989387f * fConst0);
		fConst19 = (0.0109933354f / fConst0);
		fConst20 = (0.0264077522f * fConst0);
		fConst21 = (0.361861795f / fConst0);
		fConst22 = (2953.67798f / fConst0);
		fConst23 = (880.25f / fConst0);
		iConst24 = (0.0f < (0.483715385f * fConst0));
		fConst25 = (43.3123055f * fConst0);
		fConst26 = (0.0127391135f * fConst0);
		fConst27 = (0.0230949186f / fConst0);
		fConst28 = (0.0117391134f * fConst0);
		fConst29 = (5691.92725f / fConst0);
		fConst30 = (880.0f / fConst0);
		fConst31 = (88.5032272f * fConst0);
		fConst32 = (0.00242688088f * fConst0);
		fConst33 = (0.0112993335f / fConst0);
		fConst34 = (0.00142688095f * fConst0);
		fConst35 = (1000.0f / fConst0);
		fConst36 = (52562.1992f / fConst0);
		fConst37 = (90.0463486f * fConst0);
		fConst38 = (0.00253567565f * fConst0);
		fConst39 = (0.0111057041f / fConst0);
		fConst40 = (0.00153567572f * fConst0);
		fConst41 = (50059.3984f / fConst0);
		
	}
	
	virtual void instanceResetUserInterface() {
		
	}
	
	virtual void instanceClear() {
		for (int l0 = 0; (l0 < 2); l0 = (l0 + 1)) {
			fRec1[l0] = 0.0f;
			
		}
		for (int l2 = 0; (l2 < 2); l2 = (l2 + 1)) {
			fRec3[l2] = 0.0f;
			
		}
		for (int l3 = 0; (l3 < 2); l3 = (l3 + 1)) {
			fRec6[l3] = 0.0f;
			
		}
		for (int l4 = 0; (l4 < 2); l4 = (l4 + 1)) {
			fRec7[l4] = 0.0f;
			
		}
		for (int l5 = 0; (l5 < 2); l5 = (l5 + 1)) {
			fRec4[l5] = 0.0f;
			
		}
		for (int l6 = 0; (l6 < 2); l6 = (l6 + 1)) {
			fRec9[l6] = 0.0f;
			
		}
		for (int l7 = 0; (l7 < 2); l7 = (l7 + 1)) {
			fRec10[l7] = 0.0f;
			
		}
		for (int l8 = 0; (l8 < 2); l8 = (l8 + 1)) {
			fRec12[l8] = 0.0f;
			
		}
		for (int l9 = 0; (l9 < 2); l9 = (l9 + 1)) {
			fRec13[l9] = 0.0f;
			
		}
		for (int l10 = 0; (l10 < 2); l10 = (l10 + 1)) {
			fRec15[l10] = 0.0f;
			
		}
		for (int l11 = 0; (l11 < 2); l11 = (l11 + 1)) {
			fRec17[l11] = 0.0f;
			
		}
		
	}
	
	virtual void init(int samplingFreq) {
		classInit(samplingFreq);
		instanceInit(samplingFreq);
	}
	virtual void instanceInit(int samplingFreq) {
		instanceConstants(samplingFreq);
		instanceResetUserInterface();
		instanceClear();
	}
	
	virtual mydsp* clone() {
		return new mydsp();
	}
	virtual int getSampleRate() {
		return fSamplingFreq;
		
	}
	
	virtual void buildUserInterface(UI* ui_interface) {
		ui_interface->openVerticalBox("test");
		ui_interface->closeBox();
		
	}
	
	virtual void compute(int count, FAUSTFLOAT** inputs, FAUSTFLOAT** outputs) {
		FAUSTFLOAT* output0 = outputs[0];
		for (int i = 0; (i < count); i = (i + 1)) {
			fRec1[0] = std::min<float>(fConst2, (fRec1[1] + 1.0f));
			int iTemp0 = (fRec1[0] < fConst5);
			float fRec0 = ((fRec1[0] < fConst3)?(iTemp0?((fRec1[0] < 0.0f)?0.0f:(iTemp0?(fConst6 * fRec1[0]):85.75f)):85.75f):((fRec1[0] < fConst2)?((fConst4 * (0.0f - (54.25f * (fRec1[0] - fConst3)))) + 85.75f):31.5f));
			fRec3[0] = (fConst7 + (fRec3[1] - std::floor((fConst7 + fRec3[1]))));
			fRec6[0] = std::min<float>(fConst9, (fRec6[1] + 1.0f));
			int iTemp1 = (fRec6[0] < fConst10);
			int iTemp2 = (fRec6[0] < fConst12);
			float fRec5 = (iTemp1?(iTemp2?((fRec6[0] < 0.0f)?0.0f:(iTemp2?(fConst14 * fRec6[0]):72.625f)):(iTemp1?((fConst13 * (0.0f - (4.40151501f * (fRec6[0] - fConst12)))) + 72.625f):68.2234879f)):((fRec6[0] < fConst9)?((fConst11 * (0.0f - (2.2007575f * (fRec6[0] - fConst10)))) + 68.2234879f):66.022728f));
			fRec7[0] = (fConst15 + (fRec7[1] - std::floor((fConst15 + fRec7[1]))));
			fRec4[0] = (mydsp_faustpower8_f((0.0102040814f * std::min<float>(98.0f, (iConst8?fRec5:0.0f)))) * ftbl0mydspSIG0[(((int(((65536.0f * fRec7[0]) + (131072.0f * fRec4[1]))) % 65536) + 65536) % 65536)]);
			fRec9[0] = std::min<float>(fConst17, (fRec9[1] + 1.0f));
			int iTemp3 = (fRec9[0] < fConst18);
			int iTemp4 = (fRec9[0] < fConst20);
			float fRec8 = (iTemp3?(iTemp4?((fRec9[0] < 0.0f)?0.0f:(iTemp4?(fConst22 * fRec9[0]):78.0f)):(iTemp3?((fConst21 * (0.0f - (5.5151515f * (fRec9[0] - fConst20)))) + 78.0f):72.484848f)):((fRec9[0] < fConst17)?((fConst19 * (0.0f - (72.484848f * (fRec9[0] - fConst18)))) + 72.484848f):0.0f));
			fRec10[0] = (fConst23 + (fRec10[1] - std::floor((fConst23 + fRec10[1]))));
			fRec12[0] = std::min<float>(fConst25, (fRec12[1] + 1.0f));
			int iTemp5 = (fRec12[0] < fConst28);
			float fRec11 = ((fRec12[0] < fConst26)?(iTemp5?((fRec12[0] < 0.0f)?0.0f:(iTemp5?(fConst29 * fRec12[0]):66.8181839f)):66.8181839f):((fRec12[0] < fConst25)?((fConst27 * (0.0f - (42.272728f * (fRec12[0] - fConst26)))) + 66.8181839f):24.545454f));
			fRec13[0] = (fConst30 + (fRec13[1] - std::floor((fConst30 + fRec13[1]))));
			fRec15[0] = std::min<float>(fConst31, (fRec15[1] + 1.0f));
			int iTemp6 = (fRec15[0] < fConst32);
			int iTemp7 = (fRec15[0] < fConst34);
			float fRec14 = (iTemp6?(iTemp7?((fRec15[0] < 0.0f)?0.0f:(iTemp7?(fConst36 * fRec15[0]):75.0f)):(iTemp6?((fConst35 * (0.0f - (5.30303049f * (fRec15[0] - fConst34)))) + 75.0f):69.6969681f)):((fRec15[0] < fConst31)?((fConst33 * (0.0f - (69.6969681f * (fRec15[0] - fConst32)))) + 69.6969681f):0.0f));
			fRec17[0] = std::min<float>(fConst37, (fRec17[1] + 1.0f));
			int iTemp8 = (fRec17[0] < fConst38);
			int iTemp9 = (fRec17[0] < fConst40);
			float fRec16 = (iTemp8?(iTemp9?((fRec17[0] < 0.0f)?0.0f:(iTemp9?(fConst41 * fRec17[0]):76.875f)):(iTemp8?((fConst35 * (0.0f - (5.435606f * (fRec17[0] - fConst40)))) + 76.875f):71.4393921f)):((fRec17[0] < fConst37)?((fConst39 * (0.0f - (71.4393921f * (fRec17[0] - fConst38)))) + 71.4393921f):0.0f));
			output0[i] = FAUSTFLOAT((2.08794999f * ((mydsp_faustpower8_f((0.0102040814f * std::min<float>(98.0f, (iConst1?fRec0:0.0f)))) * ftbl0mydspSIG0[(((int((65536.0f * (fRec3[0] + fRec4[0]))) % 65536) + 65536) % 65536)]) + (mydsp_faustpower8_f((0.0102040814f * std::min<float>(98.0f, (iConst16?fRec8:0.0f)))) * ftbl0mydspSIG0[(((int((65536.0f * (fRec10[0] + (mydsp_faustpower8_f((0.0102040814f * std::min<float>(98.0f, (iConst24?fRec11:0.0f)))) * ftbl0mydspSIG0[(((int((65536.0f * (fRec13[0] + (mydsp_faustpower8_f((0.0102040814f * std::min<float>(98.0f, (iConst16?fRec14:0.0f)))) * ftbl0mydspSIG0[(((int((65536.0f * (fRec13[0] + (mydsp_faustpower8_f((0.0102040814f * std::min<float>(98.0f, (iConst16?fRec16:0.0f)))) * ftbl0mydspSIG0[(((int((65536.0f * fRec13[0])) % 65536) + 65536) % 65536)])))) % 65536) + 65536) % 65536)])))) % 65536) + 65536) % 65536)])))) % 65536) + 65536) % 65536)]))));
			fRec1[1] = fRec1[0];
			fRec3[1] = fRec3[0];
			fRec6[1] = fRec6[0];
			fRec7[1] = fRec7[0];
			fRec4[1] = fRec4[0];
			fRec9[1] = fRec9[0];
			fRec10[1] = fRec10[0];
			fRec12[1] = fRec12[0];
			fRec13[1] = fRec13[0];
			fRec15[1] = fRec15[0];
			fRec17[1] = fRec17[0];
			
		}
		
	}

	
};

#endif
