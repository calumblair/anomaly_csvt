#include "stdafx.h"
#include "acceleratedAlgorithm.h"
#include "tracking.h"
#include "display.h"
#include "unsupervisedHeatmap.h"
#include "bayesianAnomalyOracle.h"
#include "clustering.h"
#include "trackingApp.h"
#define USE_SSE2
#ifdef USE_SSE2
#include <emmintrin.h>
#ifdef LINUX //must also define -msse4.1
#include <smmintrin.h>
#endif
#endif

// bayesian detection processor: see paper for details

using namespace cv;
using namespace std;

//need a signum function
inline float sgnf(float val) {
	return (0.f <= val) - (val < 0.f);
}

// linear regression controlled by parameter r. measures closeness to the mean vmean of a measurement v.
// if vmean > 0 and v>vmean, r controls how far along the x-axis that the closeness-measure will become minconf.
void BayesianAnomalyDetector::calcHamProbability(const Mat vmean, Mat& prob, float v){
	const float r_up = 7.0f;
	const float r_down = 1.5f;
	prob = Mat(vmean.size(), CV_32FC1);

	float a_v = abs(v);
	float s_v = sgnf(v);
	const float *vmp;
	float *out;

#ifdef USE_SSE2
	const __m128 absmask = _mm_set1_ps(-0.f);// -0.f = 1 << 31 apparently
	const __m128 rupv = _mm_set1_ps(r_up); //if r is 2 we can just do absmask >>1
	const __m128 rdownv = _mm_set1_ps(r_down);
	const __m128 avv = _mm_set1_ps(a_v);
	const __m128i svv = _mm_castps_si128(_mm_and_ps(_mm_set1_ps(v), absmask)); //sign of v (bit 31 set if -ve)
	const __m128 half = _mm_set_ps1(-0.5f);
#else
	float vma[4], vms[4], *out_;
#endif
	//get other control point for linear regression
	for (int i = 0; i < vmean.rows; i++){
		vmp = vmean.ptr<float>(i);
		out = prob.ptr<float>(i);
		int j = 0;
		for (j = 0; j < vmean.cols - 8; j += 8){
#ifdef USE_SSE2
			__m128 vmv, vmav, vtmp, vmsv;
			__m128 vmv2, vmav2, vtmp2, vmsv2;

			vmv = _mm_loadu_ps(vmp + j); //this is v
			vmav = _mm_andnot_ps(absmask, vmv); //abs(v)
			vmsv = _mm_and_ps(vmv, absmask); //sign vector

			vmv2 = _mm_loadu_ps(vmp + j + 4); //this is v
			vmav2 = _mm_andnot_ps(absmask, vmv2); //abs(v)
			vmsv2 = _mm_and_ps(vmv2, absmask);

			__m128 cond = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(_mm_cmpgt_ps(avv, vmav)), _mm_cmpeq_epi32(svv, _mm_castps_si128(vmsv))));
			//										  AND(								 av>avm?					sv == svm?)
			__m128 cond2 = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(_mm_cmpgt_ps(avv, vmav2)), _mm_cmpeq_epi32(svv, _mm_castps_si128(vmsv2))));

			vtmp = _mm_blendv_ps(
				_mm_min_ps(_mm_sub_ps(vmav, rdownv), _mm_mul_ps(vmav, half)),
				_mm_max_ps(_mm_mul_ps(rupv, vmav), _mm_add_ps(vmav, rupv)),
				cond);
			vtmp2 = _mm_blendv_ps(
				_mm_min_ps(_mm_sub_ps(vmav2, rdownv), _mm_mul_ps(vmav2, half)),
				_mm_max_ps(_mm_mul_ps(rupv, vmav2), _mm_add_ps(vmav2, rupv)),
				cond2);
			//select max(r*avm,r+avm) if cond is true or min(.5*avm,avm-r) if not
			vtmp = _mm_xor_ps(vtmp, vmsv);
			vtmp2 = _mm_xor_ps(vtmp2, vmsv2);
			//multiply by sign bit: res xor sign(vm)

			_mm_storeu_ps(out + j, _mm_sub_ps(vtmp, vmv));
			_mm_storeu_ps(out + j + 4, _mm_sub_ps(vtmp2, vmv2));
			//subtract original and save
#else
			vma[0] = abs(vmp[j]);
			vma[1] = abs(vmp[j + 1]);
			vma[2] = abs(vmp[j+2]);
			vma[3] = abs(vmp[j+3]);
			vms[0] = sgnf(vmp[j]);
			vms[1] = sgnf(vmp[j+1]);
			vms[2] = sgnf(vmp[j+2]);
			vms[3] = sgnf(vmp[j+3]);

			if((a_v> vma[0]) && (s_v == vms[0]))
			{ out_[j  ] = vms[0] * MAX(r_up*vma[0],vma[0]+r_up) - vmp[j  ];} else {out[j  ] = vms[0] * std::min(-0.5f*vma[0],vma[0]-r_down) - vmp[j  ];}
			if((a_v> vma[1]) && (s_v == vms[1]))
			{ out_[j+1] = vms[1] * MAX(r_up*vma[1],vma[1]+r_up) - vmp[j+1];} else {out[j+1] = vms[1] * std::min(-0.5f*vma[1],vma[1]-r_down) - vmp[j+1];}
			if((a_v> vma[2]) && (s_v == vms[2]))
			{ out_[j+2] = vms[2] * MAX(r_up*vma[2],vma[2]+r_up) - vmp[j+2];} else {out[j+2] = vms[2] * std::min(-0.5f*vma[2],vma[2]-r_down) - vmp[j+2];}
			if((a_v> vma[3]) && (s_v == vms[3]))
			{ out_[j+3] = vms[3] * MAX(r_up*vma[3],vma[3]+r_up) - vmp[j+3];} else {out[j+3] = vms[3] * std::min(-0.5f*vma[3],vma[3]-r_down) - vmp[j+3];}

			assert (out[j]   - out_[j] < 1e-6);
			assert (out[j+1] - out_[j+1] < 1e-6);
			assert (out[j+2] - out_[j+2] < 1e-6);
			assert (out[j+3] - out_[j+3] < 1e-6);
#endif
		}
		for (j;j<vmean.cols;j++){
			//get control point
			float abs_vmean = abs(vmp[j]);
			float sgn_vmean = sgnf(vmp[j]);
			if ((a_v > abs_vmean) && (s_v == sgn_vmean)){
				out[j] = sgn_vmean * std::max(r_up*abs_vmean, abs_vmean + r_up) - vmp[j];
			}
			else {
				out[j] = sgn_vmean * std::min(-0.5f*abs_vmean, abs_vmean - r_down) - vmp[j];
			}
		}
	}
	prob = (minconf - maxconf) / prob; //prob now contains the gradient
	Mat offset = maxconf - prob.mul(vmean); //c = y-m.*x

	prob = prob * v + offset; //y=mx+c
	prob = max(min(prob, maxconf), minconf); //clamp to maxconf and minconf
}

float BayesianAnomalyDetector::calcBayesianProbability(const Mat presence, const Mat ham,
	float pAnomPrior, float backgroundAnom, float strength)
{
	Mat prob, tmpA, tmpB;

	//now get bayesian measure	
	//p(a|d) = p(d|a) p(a)/ (p(d|a) p(a) + p(d|~a) p(~a)) where p(~a) = p(1-a)
	//amnd p(d|~a) is ham

	float num = pAnomPrior * backgroundAnom;

	Mat recip_den = 1.f / (num + ((1.0f - pAnomPrior) * ham));
	//now take into account areas with low object  freqs
	if (strength > 0){
		prob = (1. / (strength + presence)).mul(strength*pAnomPrior + presence.mul(recip_den, num));
	}
	else {
		prob = num * recip_den;
	}

	log(1.f - prob, tmpA);
	log(prob, tmpB);
	Scalar eta = cv::sum(tmpA - tmpB);

	float score = 1 / (1 + expf(eta[0]));
	return score;
}

void BayesianAnomalyDetector::generateRho(cv::Rect roi_, float vx, float vy, int objtype,
	float* anom_score_rho, float* anom_score_theta, float pAnomPrior, float backgroundAnom, float strength)
{
	Rect roi = clipRect(roi_, hm->origMapSize);
	Mat presence, mean_motion_x, mean_motion_y, mean_mag, ham;
	float vrho = 0;

	//get presence data
	ll->getRegionData(roi, objtype, 'x', presence);

	//do x
	hm->getRegionData(roi, objtype, 'x', mean_motion_x);
	hm->getRegionData(roi, objtype, 'y', mean_motion_y);
	magnitude(mean_motion_x, mean_motion_y, mean_mag);
	magnitude(&vx, &vy, &vrho, 1);

	calcHamProbability(mean_mag, ham, vrho); //only for x at the moment
	*anom_score_rho = calcBayesianProbability(presence, ham, pAnomPrior, backgroundAnom, strength);

	*anom_score_theta = 0;
}

void BayesianAnomalyDetector::calcAnomalyScores(cv::Rect roi_, float vx, float vy, int objtype,
	float* anom_score_x, float* anom_score_y, float pAnomPrior, float backgroundAnom, float strength)
{
	Rect roi = clipRect(roi_, hm->origMapSize);
	Mat presence, mean_motion, ham;

	//get presence data (stored in 'x' val of ll)
	ll->getRegionData(roi, objtype, 'x', presence);

	//do x
	hm->getRegionData(roi, objtype, 'x', mean_motion);
	calcHamProbability(mean_motion, ham, vx); //only for x at the moment
	*anom_score_x = calcBayesianProbability(presence, ham, pAnomPrior, backgroundAnom, strength);

	//now do y
	hm->getRegionData(roi, objtype, 'y', mean_motion);
	calcHamProbability(mean_motion, ham, vy); //only for x at the moment
	*anom_score_y = calcBayesianProbability(presence, ham, pAnomPrior, backgroundAnom, strength);
}

void BayesianAnomalyDetector::updateAll(std::list<ObjectTracker>* trackers, int age_thresh, bool alsoUpdateTrackers){
	for (list<ObjectTracker>::iterator i_kf = trackers->begin(); i_kf != trackers->end(); i_kf++)
	{
		if ((i_kf->age > age_thresh) && !i_kf->lost){
			if (alsoUpdateTrackers){ //updatetracker BEFORE we take into account the contribution of this object to heatmap
				float obj_current_anom = 0;
				float anom_x = 0, anom_y = 0;
				int id = i_kf->id;
				int factor = 5;
				//anomaly calculation goes here
				calcAnomalyScores(i_kf->bb, i_kf->statePost.at<float>(2), i_kf->statePost.at<float>(3),
					i_kf->type, &anom_x, &anom_y);
				obj_current_anom = (factor*anom_x + factor*anom_y);
				i_kf->anom += obj_current_anom;
			}
		}
	}
}
