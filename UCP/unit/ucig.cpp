// ucig.cpp
// Unit Commitment Instance Generator
// AMPL implementation by Fabrizio Lacalandra.
// c++ translation by Luigi Poderico (http://poderico.supereva.it)

#ifdef _MSC_VER
// Disable the warning n. 4786
#pragma warning( disable : 4786 )
#endif

#include<stdio.h>
#include<vector>
#include<iostream>
#include<fstream>
#include<math.h>
#include<string>
#include<map>
#include<set>
#include <sstream>
#include <cassert>
#include <climits>

using namespace std;

static double Uniform(double min, double max)
{
	double r = (double(rand())/double(RAND_MAX)) * (max-min) + min;
	return r;
}

class ThermalData
{
public:
	ThermalData() {};

	void InitPowerData(int k, int horizonLen, int gmax,int csc,double a_min, double a_max, int difficulty);

	friend ostream& operator << (ostream& s, const ThermalData& t);

	double GetMinPower() const { return fInf; }
	double GetMaxPower() const { return fSup; }
	double GetStoria(int i) const { return fStoria[i]; }
	double GetTOff() const { return fTOff; }
	double GetTOn() const { return fTOn; }

private:
	// Power units data
	int fHorizonLen;

	double fA, fB, fC;
	double fInf, fSup;
	int fTOn, fTOff;
	double fCosto, fCostof, fCostoc, fTau;
	double fTauMax, fSUCC;
	vector<int> fStoria;
	double fP0, fComb;
	double fRampaUp, fRampaDwn;

	int fSmall;
	int fMedium;
	int fBig;
};

class HydroData
{
public:
	HydroData()
		:fP_H_MAX(0)
		,fP_H_MIN(0)
		,fP_H_FLAT(0)
	{};

	void InitHydroData(int horizonLen);

	friend ostream& operator << (ostream& s, const HydroData& h);

	double GetMinPower() const { return fP_H_MIN; }
	double GetMaxPower() const { return fP_H_MAX; }
	double GetPFLAT() const { return fP_H_FLAT; }

private:
	double fa_h;
	double fb_h;

	vector<double> finflow; // {HYDRO,ORIZZONTE} default 0;

	double fdisch_max;
	double fdisch_flat;

	double fvol_init;
	double fvol_min;
	double fvol_max;

	double fP_H_MAX;
	double fP_H_MIN;
	double fP_H_FLAT;
};

class HydroCascadeData
{
public:
	void InitHydroData(int horizonLen);

	friend ostream& operator << (ostream& s, const HydroCascadeData& h);

private:
   vector<HydroData> fHydroUnits;

};

class Load
{
public:
	Load() {};

	void InitLoad(
		int horizonLen,
		int breaks,
		vector<ThermalData>& thermalData,
		vector<HydroData>& hydroData);

	friend ostream& operator << (ostream& s, const Load& l);

private:
	double fcarico_max;
	int fBreaks;
	vector<double> fcarico;
	vector<double> fPercDemand;
	vector<double> fRRperc;

	/*
	Read from a file the description of demands curve.
	*/
	void ReadPercDemand(int horizonLen);
};

class UCIG
{
private:
	// This parameteres should be read from argc/argv
	int fGg; // Days of horizone
	int fBreaks; // Subdivisions for each day
	int fGmax; // Number of thermals units
	double fa_min; //min value of the thermal quadratic coefficient
	double fa_max; //max value of the thermal quadratic coefficient
	int fImax; // Number of hydro units (hmax=0=> pure thermal system)
	int fCMax; // Number of hydro cascade units
	int fCsc; // 1 => constant start up cost
	int fdifficulty; //1,2, or 3 difficulty_level of the random data file
	int fHorizonLen; // fGg * fBreaks

	unsigned int fSeed; // The seed for the random generator number.

	double fMinPower;
	double fMaxPower;
	double fMaxThermal;

public:
	UCIG(
		int gg,
		int breaks,
		int gmax,
		double a_min,
		double a_max,
		int imax,
		int cMax,
		unsigned int seed,
		int csc,
		int difficulty
		);
	
	/*
	Initialize all the datas.
	*/
	void InitData();
	
	/*
	Write, on the standard output, the general informations:
	
	*/
	void WriteGeneralInformation();

	
	/*
	Write on the standard output the load curve.
	*/
	void WriteLoadCurve();


	
	/*
	Write on the standard output the data for thermal units.
	*/
	void WriteThermalData();


	/*
	Write on the standart output the data for hydro units.
	*/
	void WriteHydroData();
	

	/*
	Write on the standart output the data for hydro cascade units.
	*/
	void WriteHydroCascadeData();

   /*
   Update the random generator seed.
   */
   void NextSeed();

private:
	vector<ThermalData> fThermalData;
	vector<HydroData> fHydroData;
   vector<HydroCascadeData> fHydroCascadeData;
	Load fLoad;

};

//

UCIG::UCIG(
		int gg,
		int breaks,
		int gmax,
		double a_min,
		double a_max,
		int imax,
		int cMax,
		unsigned int seed,
		int csc,
		int difficulty)
  :fGg(gg)
  ,fBreaks(breaks)
  ,fGmax(gmax)
  ,fa_min(a_min)
  ,fa_max(a_max)
  ,fImax(imax)
  ,fCMax(cMax)
  ,fHorizonLen(fGg*fBreaks)
  ,fSeed(seed)
  ,fCsc(csc)
  ,fdifficulty(difficulty)
{}

//

void UCIG::InitData()
{
	fThermalData.resize(fGmax);
	fMinPower = 0;
	fMaxPower = 0;
	fMaxThermal = 0;

	srand(fSeed);

	int k;
	for (k=0; k<fGmax; ++k)
	{
		fThermalData[k].InitPowerData(k, fHorizonLen, fGmax, fCsc, fa_min, fa_max, fdifficulty);
		fMinPower += fThermalData[k].GetMinPower();
		fMaxPower += fThermalData[k].GetMaxPower();
		fMaxThermal += fThermalData[k].GetMaxPower();
	}

	fHydroData.resize(fImax);
	for (k=0; k<fImax; ++k)
	{
		fHydroData[k].InitHydroData(fHorizonLen);
		fMinPower += fHydroData[k].GetMinPower();
		fMaxPower += fHydroData[k].GetMaxPower();
	}

   fHydroCascadeData.resize(fCMax);
   for (k=0; k<fCMax; ++k)
   {
      fHydroCascadeData[k].InitHydroData(fHorizonLen);;
   }

	fLoad.InitLoad(fHorizonLen, fBreaks, fThermalData, fHydroData);

}

//

void UCIG::NextSeed()
{
   ++fSeed;
}

//

void UCIG::WriteGeneralInformation()
{
	cout << "ProblemNum\t" << fSeed << endl;
	cout << "HorizonLen\t" << fHorizonLen << endl;
	cout << "NumThermal\t" << fGmax << endl;
	cout << "NumHydro\t" << fImax << endl;
	cout << "NumCascade\t" << fCMax << endl;
}



//

void UCIG::WriteLoadCurve()
{
	cout << "LoadCurve" << endl;
	cout << "MinSystemCapacity\t" << fMinPower << endl;
	cout << "MaxSystemCapacity\t" << fMaxPower << endl;
	cout << "MaxThermalCapacity\t" << fMaxThermal << endl;

	cout << fLoad << endl;
}

//


void UCIG::WriteThermalData()
{
	cout << "ThermalSection" << endl;
	for (int k=0; k<fGmax; ++k)
	{
		cout << k << '\t' << fThermalData[k] << endl;
	}
}

//


void UCIG::WriteHydroData()
{
	cout << "HydroSection" << endl;
	for (int k=0; k<fImax; ++k)
	{
		cout << k << '\t' << fHydroData[k] << endl;
	}
}

//

void UCIG::WriteHydroCascadeData()
{
	cout << "HydroCascadeSection" << endl;
	for (int k=0; k<fCMax; ++k)
	{
		cout << k << '\t' << fHydroCascadeData[k] << endl;
	}
}

//

ostream& operator << (ostream& s, const ThermalData& t)
{
	s 	<< t.fA*t.fComb << '\t'
		<< t.fB*t.fComb << '\t'
		<< t.fC*t.fComb << '\t'
		<< t.fInf << '\t'
		<< t.fSup << '\t'
		<< t.fStoria[0] << '\t'
		<< t.fTOn << '\t'
		<< t.fTOff << '\t'
		<< t.fCostof*t.fComb << '\t'
		<< t.fCostoc*t.fComb << '\t'
		<< t.fTau << '\t'
		<< t.fTauMax << '\t'
		<< t.fCostoc*t.fCosto << '\t'
		<< t.fSUCC*t.fCosto << '\t'
		<< t.fP0;

	return s;
}

//

void ThermalData::InitPowerData(int k, int horizonLen, int gmax, int csc, double a_min, double a_max, int difficulty)
{

	fStoria.resize(horizonLen+1);
	fA = Uniform(a_min,a_max);

	fSmall = 0;
	fMedium = 0;
	fBig = 0;
	
	double diff=(a_max-a_min)/3;
	
	double a_min_small=a_max-diff;
	double a_min_medium=a_min_small-diff;
	double a_min_big=a_min_medium-diff;
	
	if (difficulty>=2){ //forces a balance with respect to the big/medium/small unit number
		if (fSmall<gmax/3)
		fA=Uniform(a_min_small,a_max);
		if (fMedium<gmax/3)
		fA=Uniform(a_min_medium,a_min_small);
		if (fBig<gmax/4)
		fA=Uniform(a_min,a_min_medium);
	}
	
	
	if (fA >=a_min_small)
	{
		switch (difficulty)
		{
		case 1:
		fTOn = int(Uniform(1, 1));
		fTOff = int(Uniform(1, 1));
		break;
		case 2:
		fTOn = int(Uniform(1, 2));
		fTOff = int(Uniform(1, 2));
		break;
		case 3:
		fTOn = int(Uniform(2, 3));
		fTOff = int(Uniform(2, 3));
		break;
		}
		
		fInf = Uniform(30, 50);
		fSup = Uniform(100, 130);
		fB = Uniform(4, 5);
		fC = Uniform(100, 150);
		fSmall++;
	}
	else if (fA>=a_min_medium && fA<=a_min_small)
	{
		switch (difficulty)
		{
		case 1:
		fTOn = int(Uniform(2, 3));
		fTOff = int(Uniform(3, 3));
		break;
		case 2:
		fTOn = int(Uniform(2, 3));
		fTOff = int(Uniform(3, 4));
		break;
		case 3:
		fTOn = int(Uniform(3, 4));
		fTOff = int(Uniform(3, 4));
		break;
		}
		
		fInf = Uniform(50, 70);
		fSup = Uniform(150, 200);
		fB = Uniform(6.7, 7.4);
		fC = Uniform(190, 350);
		fMedium++;
	}
	else
	{
		fTOn = int(Uniform(10, 14));
		fTOff = int(Uniform(10, 14));
		switch (difficulty)
		{
		case 1:
		fTOn = int(Uniform(6, 7));
		fTOff = int(Uniform(6, 7));
		break;
		case 2:
		fTOn = int(Uniform(7, 8));
		fTOff = int(Uniform(7, 8));
		break;
		case 3:
		fTOn = int(Uniform(9, 10));
		fTOff = int(Uniform(9, 10));
		break;
		}
		
		fInf = Uniform(70, 100);
		fSup = Uniform(200, 330);
		fB = Uniform(7, 8.5);
		fC = Uniform(400, 550);	
		fBig++;
	}
	
	fStoria[0] = int(Uniform(-5,5));
	if (-1<fStoria[0] && fStoria[0]<1) // fStoria[0]==0 ?
		fStoria[0] = int (pow(-1.0, k) * fTOn);

	
	
	if (csc==1)
	{
	  fCostof = 0;
	  fCostoc = 0;
	}
	else
	{
	fCostof = Uniform(200, 250);
	fCostoc = Uniform(150, 200);
	}
	fCosto = Uniform(1.3*fSup, 1.6*fSup);

	fComb = 1; //Uniform(1, 1.1);
	fTau = Uniform(1, 2);

	fP0 = (fStoria[0]>0? Uniform(1.2*fInf, 0.9*fSup): 0);
	fTauMax = 5;
	fSUCC = fCosto;

	for (int i=0; i<horizonLen; ++i)
	{
		if (fStoria[i]>=1)
			fStoria[i+1] = fStoria[i]+1;
		else
			fStoria[i+1] = (-1<fStoria[i]-1? -1: fStoria[i]-1);
	}
}

//

void HydroData::InitHydroData(int horizonLen)
{
	finflow.resize(horizonLen, 0);

	fa_h = Uniform(1, 1.05);
	fb_h = Uniform(5.1, 6.1);

	fdisch_max = Uniform(150*fa_h, 250*fa_h);
	fP_H_MAX = fa_h*fdisch_max;

	fvol_init = Uniform(0.07, 0.1) * horizonLen * fdisch_max; 
	fvol_min = Uniform(0.2, 0.3) * fvol_init;    
	fvol_max = Uniform(1.7, 2.0) * fvol_init;   

	int h;
	for (h=0; h<horizonLen; ++h)
	{
		finflow[h] = Uniform(0.10, 0.275) * fdisch_max;
	}

	fdisch_flat = fvol_init;
	for (h=0; h<horizonLen; ++h)
	{
		fdisch_flat += finflow[h];
	}
	fdisch_flat /= horizonLen;

	fP_H_FLAT = fa_h * (fdisch_flat<fdisch_max? fdisch_flat: fdisch_max);

}

//

ostream& operator << (ostream& s, const HydroData& h)
{
	s	<< h.fa_h << '\t'
		<< h.fb_h << '\t'
		<< h.fP_H_MAX << '\t'
		<< h.fdisch_max << '\t'
		<< h.fvol_init << '\t'
		<< h.fvol_min << '\t'
		<< h.fvol_max << endl;

	for (size_t i=0; i<h.finflow.size() ; ++i)
	{
		s	<< h.finflow[i];
		if (i<h.finflow.size()-1)
			s << '\t';
	}

	return s;
}


//

void HydroCascadeData::InitHydroData(int horizonLen)
{
   const int numHydro = int(Uniform(2, 5));
   this->fHydroUnits.resize(numHydro);
   int i=numHydro;
   while (i--)
   {
      this->fHydroUnits[i].InitHydroData(horizonLen);
   }
}

//

ostream& operator << (ostream& s, const HydroCascadeData& h)
{
   int numHydro = int(h.fHydroUnits.size());
   s << "CascadeLen\t" << numHydro << endl;

   for (int i=0; i<numHydro; ++i)
   {
      s << i << ' ' << h.fHydroUnits[i] << endl;
   }

   return s;
}

//


//

void Load::InitLoad(
				  int horizonLen,
				  int breaks,
				  vector<ThermalData>& thermalData,
				  vector<HydroData>& hydroData)
{
	fcarico.resize(horizonLen);
	const int kNumThermal = thermalData.size();
	const int kNumHydro = hydroData.size();
	fBreaks = breaks;
	this->ReadPercDemand(horizonLen);

	// Compute fcarico_max
	double totalMaxPower=0;
	double totalFlatPower=0;

	int i;
	for (i=0; i<kNumThermal; ++i)
	{
		if ((thermalData[i].GetStoria(0) <= -thermalData[i].GetTOff()) ||
			 (thermalData[i].GetStoria(0) >=1))
		{
			totalMaxPower += thermalData[i].GetMaxPower();
		}
	}
	for (i=0; i<kNumHydro; ++i)
	{
		totalFlatPower += hydroData[i].GetPFLAT();
	}

	fcarico_max = Uniform(
		totalMaxPower/2 + totalFlatPower,
		totalMaxPower * Uniform(0.7,0.9) + totalFlatPower);

	// Compute fcarico
	for (int h=0; h<horizonLen; ++h)
	{
		fcarico[h] = Uniform(0.950, 1) * fPercDemand[h] * fcarico_max;

		totalMaxPower = 0;
		double totalMinPower = 0;
		for (i=0; i<kNumThermal; ++i)
		{
			if ((thermalData[i].GetStoria(h) <= -thermalData[i].GetTOff()) ||
				 (thermalData[i].GetStoria(h) >=1))
			{
				totalMaxPower += thermalData[i].GetMaxPower();
			}
			if ((thermalData[i].GetStoria(h) >=1) &&
				 (thermalData[i].GetStoria(h) <= thermalData[i].GetTOn()))
			{
				totalMinPower += thermalData[i].GetMinPower();
			}
		}
		if (1.11 * fcarico[h] > totalMaxPower+totalFlatPower)
		{
			fcarico[h] = (Uniform(0.9, 0.95) * totalMaxPower)/1.09
				+ totalFlatPower;
		}

		if (fcarico[h] < totalMinPower+totalFlatPower)
		{
			fcarico[h] = Uniform(1.15,1.25) * totalMinPower+totalFlatPower;
		}
	}
}

//

void Load::ReadPercDemand(int horizonLen)
{
	ifstream parFile("perc.dat");
   if (!parFile.good())
   {
      cerr << "Error opening file \"perc.dat\"." << endl;
      abort();
   }
	vector<double> breaksDemand(fBreaks);
	fPercDemand.resize(horizonLen);
	fRRperc.resize(fBreaks);

	for (int i=0; i<fBreaks; ++i)
	{
		int ai;
		parFile >> ai >> breaksDemand[i];
		fRRperc[i] = Uniform(0.06, 0.11);
	}

	for (int j=0; j<horizonLen; ++j)
	{
		int i = j % fBreaks;
		fPercDemand[j] = breaksDemand[i] * Uniform(0.9, 1);
	}
}

//


//

ostream& operator << (ostream& s, const Load& l)
{
	s << "Loads\t" << l.fcarico.size()/l.fBreaks << '\t' << l.fBreaks << endl;
	for (int i=0; i<int(l.fcarico.size()); ++i)
	{
		s << l.fcarico[i];
		if (i%l.fBreaks<l.fBreaks-1)
			s << '\t';
		else
			s << endl;
	}

	s << "SpinningReserve\t" << l.fBreaks << endl;
	for (int j=0; j<l.fBreaks; ++j)
	{
		s << l.fRRperc[j];
		if (j<l.fBreaks-1)
			s << '\t';
	}

	return s;
}

//

typedef std::map<string, double> tParameters;
typedef std::set<string> tParametersType;

void parseParams(int argc, char *argv[], tParameters& aParameters)
{
	tParametersType myNoValue;
	tParametersType myWithValue;

	myNoValue.insert("-h");
	myNoValue.insert("--help");

	myWithValue.insert("--Ic");
	myWithValue.insert("--Gg");
	myWithValue.insert("--Breaks");
	myWithValue.insert("--Gmax");
	myWithValue.insert("--Imax");
	myWithValue.insert("--Cmax");
	myWithValue.insert("--Seed");
	myWithValue.insert("--CSC");
	myWithValue.insert("--Xml");

	myWithValue.insert("--AMin");
	myWithValue.insert("--AMax");
	myWithValue.insert("--Difficulty");

	for (int i=1; i<argc; ++i)
	{
		string myParam(argv[i]);

		if (myNoValue.find(myParam) != myNoValue.end())
		{
			aParameters[myParam] = -1;
			continue;
		}

		if (myWithValue.find(myParam) != myWithValue.end())
		{
			++i;
			if (i==argc)
			{
				cout << "Wrong parameters number!";
				abort();
			}

         ostringstream myTmpStream;
         myTmpStream << argv[i] << ends;

			istringstream myStrValue(myTmpStream.str());
			myStrValue >> aParameters[myParam];
			continue;
		}
	}
}

//

void setDefaults(tParameters& aParameters)
{
	aParameters["--Ic"]=1;
	aParameters["--Gg"]=1;
	aParameters["--Breaks"]=24;
	aParameters["--Gmax"]=10;
	aParameters["--Imax"]=10;
	aParameters["--Cmax"]=0;
	aParameters["--Seed"]=0;
	aParameters["--CSC"]=0;
	aParameters["--Xml"]=0;
	aParameters["--AMin"]=0.00001;
	aParameters["--AMax"]=0.1;
	aParameters["--Difficulty"]=1;
}

//

void usage(tParameters& aDefaultParams)
{
	cout << "Unit Commitment Instance Generator." << endl;
	cout << "Please use as:" << endl;
	cout << "ucig [-h | --help] | " << endl <<
		"[--Ic <integer>] [--Gg <integer>] [--Breaks <integer>]" << endl <<
		"[--Gmax <integer>] [--Imax <integer>] [--Cmax <integer>]" << endl <<
		"[--Seed <integer>] [--CSC <integer>] [--Xml <integer>]" << endl <<
		"[--AMin <double>] [--AMax <double>] [--Difficulty <integer>] " << endl <<
		"where:" << endl <<
		"Ic is the number of instance to be generate, 0 for MAXINT. ["
			<< aDefaultParams["--Ic"] << "]" << endl <<
		"Gg is the number of days of horizone. ["
			<< aDefaultParams["--Gg"] << "]" << endl <<
		"Breaks is the number of subdivisions for each day. ["
			<< aDefaultParams["--Breaks"] << "]" << endl <<
		"Gmax is the number of thermals units ["
			<< aDefaultParams["--Gmax"] << "]" << endl <<
		"Imax is the number of hydro units. ["
			<< aDefaultParams["--Imax"] << "]" << endl <<
		"Cmax is the number of hydro cascade units. ["
			<< aDefaultParams["--Cmax"] << "]" << endl <<
		"Seed is the seed for the random generator number. ["
			<< aDefaultParams["--Seed"] << "]" << endl <<
		"CSC==1 for constant start up cost. ["
			<< aDefaultParams["--CSC"] << "]" << endl <<
		"Xml==1 the produced result is in xml format [default Xml<>1] ["
			<< aDefaultParams["--Xml"] << "]" << endl <<
		"AMin is the minimum value of the quadratric coefficient for thermal units:tipical range [0.0001,0.0005] ["
			<< aDefaultParams["--AMin"] << "]" << endl <<
		"AMax is the maximum value of the quadratric coefficient for thermal units:tipical range [0.001,0.05] ["
			<< aDefaultParams["--AMax"] << "]" << endl <<
		"Difficulty is the *Difficulty-Level* (1 or 2 or 3) of the random data file ["
			<< aDefaultParams["--Difficulty"] << "]"<< endl;
}

//

int main(int argc, char *argv[])
{
	tParameters myParameters;

	setDefaults(myParameters);
	parseParams(argc, argv, myParameters);
	
	//cout << "argc:" <<argc<< endl;
	if (myParameters.find("-h")!=myParameters.end() ||
		 myParameters.find("--help")!=myParameters.end())
	{
		usage(myParameters);
		return 0;
	}

	int ic = int(myParameters["--Ic"]);
	int gg = int(myParameters["--Gg"]);
	int breaks = int(myParameters["--Breaks"]);
	int gmax = int(myParameters["--Gmax"]);
	double a_min = myParameters["--AMin"];
	double a_max = myParameters["--AMax"];
	int imax = int(myParameters["--Imax"]);
	int cmax = int(myParameters["--Cmax"]);
	unsigned int seed = int(myParameters["--Seed"]);
	int csc = int(myParameters["--CSC"]);
	int difficulty = int(myParameters["--Difficulty"]);

	if (a_min <=1e-12 || a_max<=1e-12)
	{
		cout << "\nERROR:a_min AND a_max MUST BE STRICTLY POSITIVE" << endl;
		cout << "a_min= " <<  a_min << " a_max= " << a_max << endl; 	
		abort();
	}
	
	if (a_min>=a_max)
	{
		cout << "\ncannot set: a_min>=a_max" << endl;
		abort();
	}
	
	if (a_min < 0.00001 || a_min>=0.01)
	{
		cout << "\nERROR:a_min= " << a_min << " is not a realistic value!!" << endl;
		cout << "tipical range for a_min is [0.0001,0.0005]"<< endl;
		abort();
	}
	if (a_max < 0.1 || a_max>=0.5)
	{
		cout << "\nERROR:a_max= " << a_max << " is not a realistic value!!" << endl;
		cout << "tipical range for a_max is [0.001,0.05]"<< endl;
		abort();
	}
	
	if (difficulty!=1 && difficulty!=2 && difficulty!=3)
	{
		cout << "\nERROR: difficulty can only be set to 1 or 2 or 3"  << endl;
		abort();
	}

	UCIG ucig(gg, breaks, gmax, a_min, a_max, imax, cmax, seed, csc, difficulty);
	
	if (ic==0)
	{
		ic = INT_MAX;
	}
	
	while (ic--)
	{
		ucig.InitData();

		ucig.WriteGeneralInformation();
		ucig.WriteLoadCurve();
		ucig.WriteThermalData();
		ucig.WriteHydroData();
		ucig.WriteHydroCascadeData();

		ucig.NextSeed();
	}
	return 0;
}

