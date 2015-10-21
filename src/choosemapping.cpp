//choosemapping.cpp
//calum blair 20/12/12
//THIS FILE MUST BE COMPILED WITH OPTIMISATION OFF OTHERWISE
//REALEASE MODE RUNS WILL FAIL INSIDE ExhaustiveSearcher::walk_internal()
//to do this you must now also turn off precompiled headers for this file
//The program appears to lose track of the data inside the exhaustivesearcher object
//especially 'valid'
//I think this is something to do with boost::multi_array thing
//but haven't investigated fully
//TODO fix this or swap the multi_array for an old-school C array

#include "stdafx.h"
#include <boost/multi_array.hpp>
#include "acceleratedAlgorithm.h"

//boost classes for matrix of implementation pointers
typedef boost::multi_array<Implementation*, 2> impl_array_type;
//typedef impl_array_type::index index;


//characteristics for priorities
//the C way
#define IMPL_TIME		1
#define IMPL_POWER		2
#define IMPL_ACCURACY	3
//the c++ way
const std::map<int, std::string> characteristics;

//a populated solution is based on an implementation with
//a vector containing pointers to the actual implementations
//which do the work
struct PopulatedSolution : public Implementation
{
	float cost; //overall cost of solution for given priorities
	std::vector<Implementation*> stages; //discrete algorithms within this solution
};

/////////////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

//exhaustive search class where the solutions are obtained recursively
class ExhaustiveSearcher{
public:
	ExhaustiveSearcher();
	ExhaustiveSearcher(int nFunctions, vector<int> nCandidates, bool debug = false);
	void walk(impl_array_type& M_); //walk through a function/implmentation matrix and build all solutions
	std::vector<std::vector<int> > getSolutions(void) //print and return solution indices
	{
		if (debug){
			for (unsigned int i = 0; i < solutions.size(); i++){
				for (unsigned int j = 0; j < solutions[i].size(); j++)
					cout << solutions[i][j] << " ";
				cout << "\n";
			}
		}
		return solutions;
	};
protected:
	void walk_internal(unsigned int x, unsigned int y, impl_array_type& M_, int dbg_depth = 0);
	//i would like to initialise a local pointer to M, the matrix of implementation pointers
	//based on http://pinyotae.blogspot.co.uk/2009/06/note-on-boost-multidimensional-array.html
	// this is possible in a function call
	//it doesnt look like its possible to declare a pointer to a multi array inside this class however,
	//because as soon as you make a pointer to a multi_array then access a specific element
	//you lose the ability to access what's actually in that element and only get the boost wrappers. hmm.

	unsigned int nx; //num of functions 
	std::vector<int> ny; //num of candidates per function.this changes from column to column

	std::vector<int> live; //this holds the current solution being built
	std::vector<std::vector<int> > solutions;
	bool debug;
};

ExhaustiveSearcher::ExhaustiveSearcher(){
	; //default constructor shouldnt get called
}
ExhaustiveSearcher::ExhaustiveSearcher(int nFunctions, vector<int> nCandidates, bool debug_){
	nx = nFunctions;
	ny = nCandidates;
	live = vector<int>(nx, 0);//preallocate
	debug = debug_;
}
void ExhaustiveSearcher::walk(impl_array_type& M_){
	walk_internal(0, 0, M_, 0);
}
void ExhaustiveSearcher::walk_internal(unsigned int x, unsigned int y, impl_array_type& M_, int dbg_depth){
	int impID = 0;
	if (debug)
		cout << dbg_depth << " walking from x " << x << ", y "
		<< y << "to x " << nx - 1 << " y " << ny[x] - 1 << "\n";
	for (y; y < ny[x]; y++)
	{ //walk thru all rows in this column vertically
		//store an index to the current implementation
		live[x] = y;
		if (debug){
			impID = M_[x][y]->id;
			cout << dbg_depth << " x " << x << " y " << y << " id " << impID << "\n";
		}

		if (x < (nx - 1)) //walk thru other columns horizontally
			walk_internal(x + 1, 0, M_, dbg_depth + 1);
		else //when (x==(nx-1)), //last column
		{
			//check that all the candidates in the solution are valid
			//and anything else, eg 2 fpga-exclusive implementations are not running
			bool valid = true;
			unsigned int fpgaRequests = 0;
			for (size_t i = 0; i < nx; i++){
				Implementation* stage = M_[i][live[i]];
				fpgaRequests += stage->isFpgaExclusive;
				valid = (valid && stage->valid);
				if (debug) cout << dbg_depth << " valid is " << valid
					<< " at i = " << i << " x " << x << ", y " << y << "\n";
			}
			if (valid && fpgaRequests < 2){
				solutions.push_back(live);
				if (debug) cout << dbg_depth << " pushing back x " << x << ", y " << y << "\n";
			}

			else{
				if (debug){
					cout << "rejecting solution ";
					for (size_t i = 0; i < nx; i++) cout << live[i] << " ";
					cout << "because valid is " << valid
						<< " (must be true) and fpgaRequests is " << fpgaRequests << " (max 1)" << "\n";
				}
			}
		}
	}
}


void getCosts(impl_array_type& M, vector <vector<int> >s, simplepriorities p,
	vector<PopulatedSolution>& costedSolutions, bool debug = false)
{
	size_t i = 0, j = 0;

	for (; i < s.size(); i++) //loop over s.size()
	{
		PopulatedSolution curr; //current (WIP) solution
		curr.resources = 0;
		curr.valid = true;
		curr.accuracy = 1;
		curr.time = 0;
		curr.power = 0;
		int fpgaRequests = 0;
		float runningenergy = 0;

		for (j = 0; j < s[i].size(); j++){
			curr.stages.push_back(M[j][s[i][j]]);
		}

		if (1){ //check everything is valid, set other fields within curr
			for (j = 0; j < curr.stages.size(); j++){
				curr.valid = curr.valid && curr.stages[j]->valid;
				curr.resources = curr.resources | curr.stages[j]->resources;

				fpgaRequests += curr.stages[j]->isFpgaExclusive;

				//there are two ways to obtain figures for acc/time/power for whole system
				//model it or learn it from a previous run
				//we're going to do a simple model of it at the moment
				//combine time - this one is easy
				curr.time += curr.stages[j]->time;

				//combine power (not just max but average power over each section)
				//curr.power = max(curr.power,curr.stages[j]->power);
				runningenergy += curr.stages[j]->time*curr.stages[j]->power;

				//combine accuracy or error, using a ranking associated with each implementation
				if (curr.stages[j]->accuracy >0)
					curr.accuracy *= curr.stages[j]->accuracy;
			}
			curr.power = runningenergy / curr.time;
			assert(fpgaRequests <= 1);
			curr.isFpgaExclusive = (fpgaRequests != 0); //->bool

			curr.algorithm = ALGORITHM_SOLUTION;
			curr.cost = curr.accuracy * p.accuracy +
				curr.time * p.latency +
				curr.power * p.power;
			curr.id = 99;
		}
		if (curr.valid)
			costedSolutions.push_back(curr);
	}
}

void chooseBestMappingFromAlgorithm(std::vector<Ptr<AcceleratedAlgorithm> > candidates,
	std::vector<int> functions, simplepriorities p,
	std::vector<Ptr<AcceleratedAlgorithm> >* mapping, Implementation* chosenSolution)
{
	//do exhaustive search here as the combination is relatively small
	unsigned int i = 0, j = 0;
	bool debug = false;
	size_t nF = functions.size();
	if (nF == 0){ //if not asked to allocate any functions
		mapping->clear();
		if (debug) cout << "zero functions this frame\n";
		chosenSolution->time = 0;
		chosenSolution->power = 0;
		chosenSolution->accuracy = 0;
		return;
	}

	std::vector<int> maxImpls(nF, 0); //keeps track of max number of implementations per function (or per-column column height)

	//make a 2-D array: dim1 is functions, dim2 is implementations
	//for large candidate sizes we could maybe make this on a column by column basis
	impl_array_type M(boost::extents[functions.size()][candidates.size()]);

	//now zero the whole thing
	for (i = 0; i < functions.size(); i++)
		for (j = 0; j < candidates.size(); j++)
			M[i][j] = 0;


	//for each function (cf algorithm) 
	//populate the array M with implementations for that function
	for (i = 0; i < functions.size(); i++){
		int k = 0;
		for (j = 0; j < candidates.size(); j++){
			if (candidates[j]->implementation.algorithm == functions[i]){
				M[i][k] = &(candidates[j]->implementation);
				k++;
			}
		}
		maxImpls[i] = k;
	}

	if (debug){//dump M
		for (i = 0; i < M.size(); i++){
			for (j = 0; j < maxImpls[i]; j++)
				cout << M[i][j]->id << " ";
			cout << "\n";
		}
	}
	//this is similar to the MATLAB exhaustiveSearch
	//we now have a matrix of possible implementations for each function
	ExhaustiveSearcher exh = ExhaustiveSearcher(nF, maxImpls, debug);
	//walk through the matrix
	exh.walk(M);
	//and return a vector of indices to possible solutions in M
	//		note we check if these are valid before we return them
	vector <vector<int> > s = exh.getSolutions();

	if (debug){	//quick "how do i access M with s" check	
		cout << "Solution matrix\n";
		for (i = 0; i < s.size(); i++){
			for (j = 0; j < s[i].size(); j++){
				cout << M[j][s[i][j]]->id << " ";
			}
			cout << "\n";
		}
	}

	vector<PopulatedSolution> C;
	//for every possible solution
	//get costs for all solutions
	//pass it the implementation matrix, solution index, and priorities
	//calculate cost (inc priorities) and push back into a solutions vector C

	//choose the solution with lowest cost
	float minCost = 0; unsigned int minCostIndex = 0;

	//construct a mapping around the populatedSolution and return it
	mapping->clear();

	if (s.size() > 0){
		getCosts(M, s, p, C, debug);
		minCost = C[0].cost;
		for (i = 0; i < C.size(); i++){
			if (C[i].cost < minCost){
				minCost = C[i].cost;
				minCostIndex = i;
			}
		}
		if (debug)
			cout << "using solution " << minCostIndex << " with cost " << minCost << "\n";

		PopulatedSolution best = C[minCostIndex];
		chosenSolution->time = best.time;
		chosenSolution->power = best.power;
		chosenSolution->accuracy = best.accuracy;

		for (i = 0; i < best.stages.size(); i++){ //for each implementation in best
			int id = best.stages[i]->id; //get its id
			for (j = 0; j < candidates.size(); j++){
				if (id == candidates[j]->getID()){ //find that AcceleratedAlgorithm in candidates
					mapping->push_back(candidates[j]); //add to mapping
					if (debug)
						cout << "adding algorithm " << candidates[j]->getID(id)
						<< " with id " << id << " to mapping\n";
					continue; //jump inner for loop
				}
			}
		}
	}
	else {
		if (debug) cout << "no valid solutions found\n";
		chosenSolution->time = 0;
		chosenSolution->power = 0;
		chosenSolution->accuracy = 0;
	}
}


//group rectangles according to the algorithm shown in mergedetections.m
void myRectangleGroup(vector<Detection>* src, vector<Detection>* dst, Size imsz){
	int maxdets = src->size();
	int i = 0, j;
	int changes = 1;
	int ix, iy, iw, ih; //bb for box i
	int x0, x1, y0, y1; //surround for box i
	int jx, jy, jw, jh; //bb for box j
	int envelope = 40; //num pixels inside which 2 bboxes will be merged

	//first. copy src to dst
	*dst = *src;
	while (changes != 0){
		changes = 0;
		while (i < maxdets){
			//get  i'th box
			ix = (*dst)[i].bb.x;
			iy = (*dst)[i].bb.y;
			iw = (*dst)[i].bb.width;
			ih = (*dst)[i].bb.height;
			//get surround for i
			x0 = max(1, ix - min(iw, envelope));
			x1 = min(imsz.width, ix + iw - min(iw, envelope));
			y0 = max(1, iy - min(ih, envelope));
			y1 = min(imsz.height, iy + ih - min(ih, envelope));

			//now look at box j
			j = i + 1;
			while (j < maxdets){ //if bbox j inside envelope of bbox i
				if ((*dst)[j].bb.x >= x0 && ((*dst)[j].bb.x + (*dst)[j].bb.width) <= x1 // a && b && c &&d all true
					&& (*dst)[j].bb.y >= y0 && ((*dst)[j].bb.y + (*dst)[j].bb.height) < y1)
				{
					//what's box j? also update box i with the new results
					(*dst)[i].bb.x = jx = min(ix, (*dst)[j].bb.x);
					(*dst)[i].bb.y = jy = min(iy, (*dst)[j].bb.y);
					(*dst)[i].bb.width = jw = max(ix + iw, ((*dst)[j].bb.x + (*dst)[j].bb.width)) - jx;
					(*dst)[i].bb.height = jh = max(iy + ih, ((*dst)[j].bb.y + (*dst)[j].bb.height)) - jy;
					(*dst)[i].score = 0.8;
					(*dst)[i].source = 99;

					//now update box i again
					ix = (*dst)[i].bb.x;
					iy = (*dst)[i].bb.y;
					iw = (*dst)[i].bb.width;
					ih = (*dst)[i].bb.height;
					//get surround for i
					x0 = max(1, ix - min(iw, envelope));
					x1 = min(imsz.width, ix + iw - min(iw, envelope));
					y0 = max(1, iy - min(ih, envelope));
					y1 = min(imsz.height, iy + ih - min(ih, envelope));

					//now swap j and the last element, so we iterate over fewer things
					(*dst)[j] = (*dst)[maxdets - 1];
					maxdets--;
					changes++;
				}
				j++;
			}
			i++;
		}
		i = 0;
	}
	//resize dst before return
	dst->resize(maxdets);
	if (dst->size() != src->size())
		cout << "Reduced " << src->size() << " detections to " << dst->size() << "\n";
}
