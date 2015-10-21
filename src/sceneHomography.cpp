#include "stdafx.h"
#include <algorithm>
using  namespace cv;
using namespace std;

//return a pre-calculated homography matrix for _each_ individual clip. gives more accurate results than learning
//a single matrix for each scene, as the camera moves about so much...
void getSceneSpecificData(const int scene_id, cv::Mat* H_cam2base, cv::Size* canvas_size,
	cv::Point* canvas_tl, cv::Point* canvas_br)
{
	*canvas_size = Size(550, 1040); //again, specific to pv3. just do all these here.
	*canvas_tl = Point(20, 15); *canvas_br = Point(400, 1040);
	switch (scene_id){
	case 3: //ilids scene 3
		*H_cam2base = *(Mat_<float>(3, 3) <<
			0.866886399411012, 2.46157940856413, -295.657124345592,
			-0.174010297130362, 11.4258430499553, -1000.66967188594,
			-0.00016016961681842, 0.00897594881264994, 0.215968887690754);
		*canvas_size = Size(550, 1040); //again, specific to pv3
		*canvas_tl = Point(20, 15); *canvas_br = Point(400, 1040);
		break;
	case 2: //ilids scene 2
		/*H_cam2base = *(Mat_<float>(3,3) <<
		0.866886399411012,		2.46157940856413,		-295.657124345592,
		-0.174010297130362,		11.4258430499553,		-1000.66967188594,
		-0.00016016961681842,	0.00897594881264994,	0.215968887690754 );
		canvas_size = Size(550,1040);
		canvas_tl = Point(20,15); canvas_br = Point(400,1040);*/
		throw(std::runtime_error("pv2 homography not implemented yet"));
		break;

	case 96:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b
			1.0093563695783, 2.79447588876782, 12.595586167071,
			-0.177161832442956, 11.0241893893402, -217.504055020933,
			-0.000299740266018032, 0.00647947571549683, 0.859594917365005);
		*canvas_size = Size(930, 1300);
		*canvas_tl = Point(0, 0); *canvas_br = Point(900, 1290);
		break;
	case 97:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b
			1.01530332951826, 2.75530695740986, 155.68439481254,
			-0.0603860218047633, 10.7949130854208, 195.114709839423,
			-0.000198011352072563, 0.00638921759629675, 1.09881346065109);//<--FIXME
		*canvas_size = Size(930, 1300);
		*canvas_tl = Point(0, 0); *canvas_br = Point(900, 1290);
		break;

	case 98: //bank st whatever
		*H_cam2base = *(Mat_<float>(3, 3) <<
			0.666279443431517, 0.136915083954275, 63.64125941039,
			0.342203997002826, 4.56041880094062, -553.467132580166,
			0.00015141427705968, 0.00363011543164001, 0.544636209505631);
		*canvas_size = Size(700, 800);
		*canvas_tl = Point(57, 10); *canvas_br = Point(612, 766);
		break;
	case 99: //bank_st_20130724
		*H_cam2base = *(Mat_<float>(3, 3) <<
			1.06149619622251, 1.6774446482998, -719.715432249837,
			0.973346495176826, 6.60741677133783, -2485.62870176156,
			0.000627386881081684, 0.00396186725015439, -0.495170702814109);
		*canvas_size = Size(740, 1300);
		*canvas_tl = Point(200, 200); *canvas_br = Point(740, 1300);
		break;
	case 101:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtrn301a
			0.954938738222607, 2.61846333071445, -398.573750294333,
			-0.13342443001249, 12.1250137590832, -1158.99523352072,
			-8.78808095323874e-005, 0.00952283238406544, 0.0871910845210797);
		break;
	case 102:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtrn301b
			0.902533276153987, 2.53373768565128, -324.14551933386,
			-0.199680590711104, 11.7714327384605, -949.148225489026,
			-0.000144875955375129, 0.00923458229288552, 0.253909729280198);
		break;
	case 103:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtrn301c
			0.928386384918475, 2.52254255838822, -296.705325102317,
			-0.0675095097213521, 11.7214093830123, -962.395954024501,
			-5.06758702575208e-005, 0.00918326920627454, 0.24578965648917);
		break;
	case 104:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtrn301d
			0.941330358179254, 2.51305147123756, -342.358070899977,
			-0.0344079594562231, 11.6701171700935, -1027.07569069787,
			1.04236485824743e-005, 0.00912815673915725, 0.191870563840878);
		break;
	case 105:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtrn301e
			1.01482013075531, 2.68615351301194, -488.675581018396,
			-0.00439759143527328, 12.3981026523412, -1454.33221650221,
			5.09623372200484e-005, 0.0096991275465694, -0.147031556543512);
		break;

	case 111:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtrn302a
			1.07879919482269, 2.80360165902049, -515.478937883422,
			0.197493994022155, 13.0969322828932, -1714.89933669117,
			0.000155045638942955, 0.01031727003888, -0.350856861706796);
		break;
	case 112:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtrn302b
			1.12020042106522, 2.81708987122537, -531.173341395267,
			0.321193435127595, 13.142301697441, -1770.38519121279,
			0.000286351250051018, 0.0103350500256895, -0.397095050063335);
		break;
	case 113:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtrn302c
			1.03934765447345, 2.6126008777416, -390.188576530608,
			0.280533742171502, 12.2268384545593, -1222.14693057251,
			0.000250613339392197, 0.00961142530974558, 0.0353152475041103);
		break;

	case 121:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtea301a
			0.986149016388256, 2.47664824655667, -310.441539906812,
			-0.0251423155802591, 11.5544066245494, -1172.04560859945,
			2.25292016997561e-005, 0.00903850323962084, 0.0806402368660759);
		break;
	case 122:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtea301b
			0.965885455326761, 2.47891747918585, -378.479358676631,
			-0.0733466717101547, 11.4941121898426, -1147.07099475754,
			-3.72681211010543e-005, 0.00902911780411179, 0.0962092746674537);
		break;
	case 123:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtea301c
			0.931495848458269, 2.36750394087031, -338.581421597195,
			-0.0505978366983447, 11.0143664730597, -1166.03628034131,
			-3.3369110789272e-005, 0.00862348132046798, 0.086492007683697);
		break;
	case 124:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtea301d
			0.966876108686232, 2.48502892420598, -313.365273272567,
			-0.0671513053118623, 11.6067526220974, -1195.63358799635,
			-2.49828801908918e-005, 0.0090834885182989, 0.0626805048949359);
		break;
	case 125:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvtea301e
			1.1614681238016, 2.66646557926157, -518.736402340373,
			0.389804519622402, 12.4620511273175, -1684.24389293017,
			0.000369588223848241, 0.00977966627884316, -0.331075484206592);
		break;

	case 131:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvten301a
			1.02441168313392, 2.57251572788894, -460.880848699962,
			-0.0143615292112344, 11.915875472246, -1392.02101666504,
			3.75035121680271e-005, 0.00935177393142024, -0.100092832160579);
		break;
	case 132:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvten301b
			0.945977557615359, 2.45309242960507, -369.180582804849,
			-0.134602782758129, 11.396598747688, -1110.24367356969,
			-9.28960273985177e-005, 0.00893479548954069, 0.127895556511443);
		break;
	case 133:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvten301c
			0.989699875838914, 2.45943457662075, -343.218919559092,
			0.0186140259471385, 11.4489214039064, -1132.1241324732,
			2.54990924745124e-005, 0.00897907060263185, 0.111000455659037);
		break;
	case 134:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvten301d
			0.971102849855745, 2.47244337713542, -387.392223959555,
			-0.0528858920930099, 11.4850142735316, -1191.19102888462,
			-1.76514451118086e-005, 0.00900899039400303, 0.0624379583043444);
		break;
	case 135:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvten301e
			1.02984790115131, 2.63164905769165, -520.255087189789,
			-0.0530105213227721, 12.2053715259773, -1596.64873464576,
			-3.8125800691937e-006, 0.00958322698286365, -0.26002273116799);
		break;

	case 141:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvten302a
			1.13694087303883, 2.73380485589075, -569.165112616461,
			0.233494987324594, 12.7539288066292, -1887.57495171049,
			0.00019448354495107, 0.0100422694178099, -0.48786062833085);
		break;
	case 142:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvten302b
			1.1351114393736, 2.65013066709661, -515.53915848418,
			0.318590890952864, 12.4068474530068, -1679.88501820412,
			0.000275036158065227, 0.00975127961471357, -0.323939803842279);
		break;
	case 143:
		*H_cam2base = *(Mat_<float>(3, 3) << // c2b_pvten302c
			1.23017148878006, 2.7408793810089, -594.706058648983,
			0.527726554685807, 12.8443590570664, -2016.72788129036,
			0.00049955993577726, 0.0100891238156779, -0.596626590038355);
		break;
	default:
		throw(std::runtime_error("unknown scene, no homography data known"));
		break;
	}
}

int getHomographyIndex(std::string token_){
	//got to do this in two passes
	int lastdot = token_.find_last_of(".");
	string token = token_.substr(0, lastdot);
	int lastdir = max((int)token.find_last_of("\\") + 1, 0); //bug magnet! TODO just use boost
	token = token.substr(lastdir, token.npos);

	std::transform(token.begin(), token.end(), token.begin(), ::tolower);

	if (token == "pvtrn301a")
		return 101;
	else if (token == "pvtrn301b")
		return 102;
	else if (token == "pvtrn301c")
		return 103;
	else if (token == "pvtrn301d")
		return 104;
	else if (token == "pvtrn301e")
		return 105;

	else if (token == "pvtrn302a")
		return 111;
	else if (token == "pvtrn302b")
		return 112;
	else if (token == "pvtrn302c")
		return 113;

	else if (token == "pvtea301a")
		return 121;
	else if (token == "pvtea301b")
		return 122;
	else if (token == "pvtea301c")
		return 123;
	else if (token == "pvtea301d")
		return 124;
	else if (token == "pvtea301e")
		return 125;

	else if (token == "pvten301a")
		return 131;
	else if (token == "pvten301b")
		return 132;
	else if (token == "pvten301c")
		return 133;
	else if (token == "pvten301d")
		return 134;
	else if (token == "pvten301e")
		return 135;

	else if (token == "pvten302a")
		return 141;
	else if (token == "pvten302b")
		return 142;
	else if (token == "pvten302c")
		return 143;

	else if (token == "bankst_mvi_2378_720_train")
		return 97;
	else if (token == "bankst_mvi_2378_720_test")
		return 97;
	else if (token == "bankst_MVI_0016")
		return 96;
	else if (token == "mvi_0016_car1")
		return 96;
	else if (token == "mvi_0016_ped1")
		return 96;

	else
		throw(std::runtime_error(" warning: learned homography not found!\n"));
}
