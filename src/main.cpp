#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <iomanip>
#include <stdexcept>
//#include <armadillo>
#define K 0.2 // convergence speed
#define Inte_distance 10 // convergence speed
#define Segment 10e3 // convergence speed
#define tolerance 0.01 // convergence speed
#define _USE_MATH_DEFINES
#define normal_factor_mod 2.0
#define normal_factor_IO 30
#define step_size 5  //conjugate gradient step
#define boundary_force 100

using namespace std;

const double NEARZERO = 1.0e-5;
//using vector<double> = vector<double>;         // vector
//using vector<vector<double>> = vector<vector<double>>;            // vector<vector<double>> (=collection of (row) vectors)

// Prototypes for conjugate gradient method-------------------------------------------------------------------
void print(string title, const vector<double> &V);
void print(string title, const vector<vector<double>> &A);
vector<double> matrixTimesVector(const vector<vector<double>> &A, const vector<double> &V);
vector<double> vectorCombination(double a, const vector<double> &U, double b, const vector<double> &V);
double innerProduct(const vector<double> &U, const vector<double> &V);
double vectorNorm(const vector<double> &V);
vector<double> conjugateGradientSolver(const vector<vector<double>> &A, const vector<double> &B);

// Function details for conjugate gradient method-------------------------------------------------------------------
void print(string title, const vector<double> &V)
{
	cout << title << '\n';

	int n = V.size();
	for (int i = 0; i < n; i++)
	{
		double x = V[i];   if (abs(x) < NEARZERO) x = 0.0;
		cout << x << '\t';
	}
	cout << '\n';
}



void print(string title, const vector<vector<double>> &A)
{
	cout << title << '\n';

	int m = A.size(), n = A[0].size();                      // A is an m x n vector<vector<double>>
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double x = A[i][j];   if (abs(x) < NEARZERO) x = 0.0;
			cout << x << '\t';
		}
		cout << '\n';
	}
}

vector<double> matrixTimesVector(const vector<vector<double>> &A, const vector<double> &V)     // Matrix times vector
{
	int n = A.size();
	vector<double> C(n);
	for (int i = 0; i < n; i++) C[i] = innerProduct(A[i], V);
	return C;
}

vector<double> vectorCombination(double a, const vector<double> &U, double b, const vector<double> &V)        // Linear combination of vectors
{
	int n = U.size();
	vector<double> W(n);
	for (int j = 0; j < n; j++) W[j] = a * U[j] + b * V[j];
	return W;
}

double innerProduct(const vector<double> &U, const vector<double> &V)          // Inner product of U and V
{
	return inner_product(U.begin(), U.end(), V.begin(), 0.0);
}

double vectorNorm(const vector<double> &V)                          // Vector norm
{
	return sqrt(innerProduct(V, V));
}

vector<double> conjugateGradientSolver(const vector<vector<double>> &A, const vector<double> &B)
{
	double TOLERANCE = 1.0e-10;

	int n = A.size();
	vector<double> X(n, 0.0);

	vector<double> R = B;
	vector<double> P = R;
	int k = 0;

	while (k < n)
	{
		vector<double> Rold = R;                                         // Store previous residual
		vector<double> AP = matrixTimesVector(A, P);

		double alpha = innerProduct(R, R) / max(innerProduct(P, AP), NEARZERO);
		X = vectorCombination(1.0, X, alpha/step_size, P);            // Next estimate of solution
		R = vectorCombination(1.0, R, -alpha/ step_size, AP);          // Residual 

		if (vectorNorm(R) < TOLERANCE) break;             // Convergence test

		double beta = innerProduct(R, R) / max(innerProduct(Rold, Rold), NEARZERO);
		P = vectorCombination(1.0, R, beta/ step_size, P);             // Next gradient
		k++;
	}

	return X;
}
//

//conjugate gradient solver 2.0
vector<double> conjugate_gradient(const vector<vector<double> > & A, const vector<double> & b, int T)
{
	int N = b.size();
	vector<double> r(N, 0.0);
	vector<double> p(N, 0.0);
	vector<double> x(N, 0.0);
	for (int i = 0; i < N; i++)
		p[i] = r[i] = b[i];
	int t = 0;
	while (t < T)
	{
		double rtr = 0.0;
		double ptAp = 0.0;
		for (int i = 0; i < N; i++)
			rtr += r[i] * r[i];
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				ptAp += A[i][j] * p[i] * p[j];
		double alpha = rtr / (ptAp + 1e-10);
		vector<double> rn(N, 0.0);
		for (int i = 0; i < N; i++)
		{
			x[i] += alpha * p[i];
			rn[i] = r[i];
			for (int j = 0; j < N; j++)
				rn[i] -= alpha * A[i][j] * p[j];
		}
		double rntrn = 0.0;
		for (int i = 0; i < N; i++)
			rntrn += rn[i] * rn[i];
		if (rntrn < 1e-10) break;
		double beta = rntrn / rtr;
		for (int i = 0; i < N; i++)
		{
			p[i] = beta * p[i] + rn[i];
			r[i] = rn[i];
		}
		t++;
	}
	return x;
}


//

//class definition-------------------------------------------------------------------
class module {
public:
	module() { x = 0; y = 0; }
	string name;
	double width;
	double height;
	double x; //left most 
	double y; //bottom most
	double x_cent; // (x+w)/2
	double y_cent;
	int idx_i, idx_j; //index in p, d, C 

	void init_xy_cent() {
		x_cent = (x + width) / 2.0;
		y_cent = (y + height) / 2.0;
		return;
	}

	void moveto(double x_coord, double y_coord) {
		x = x_coord - width/2.0;
		y = y_coord - height/2.0;
		x_cent = x_coord;
		y_cent = y_coord;
		return;
	}

};

class net {
public:
	net(string name) { this->name = name; }
	//vector<string> sb;
	//vector<string> IO;	
	vector<module*> sb;
	vector<module*> IO;
	string name;

	int get_pinNum() {
		return sb.size() + IO.size();
	}

};
//

//global var
vector<module> modlist;
vector<module> IOlist;
vector<net> netlist;
int grid_h, grid_v;
double h_cent, v_cent;
double coef_k;
int numMod, numIO;
int totalCell;
int numNet;

vector<double> p, d, e;
vector<double> cost;
vector<vector<double>> C;
vector<vector<double>> density_map;
//

//parser
void parse_module(string file_path_module) {
	
	ifstream input;
	string dummy;
	input.open(file_path_module);
	if (!input.is_open()) {
		cout << "Failed to load *.module" << endl;
		return;
	}
	input >> dummy >> grid_h >> grid_v;
	h_cent = grid_h / 2.0;
	v_cent = grid_v / 2.0;
	input >> dummy >> numMod;
	input >> dummy >> numIO;


	//fscanf(input_module, "grid %d %d\n", &grid_h, &grid_v);
	//h_cent = grid_h / 2.0;
	//v_cent = grid_v / 2.0;
	//fscanf(input_module, "NumModules %d\n", &numMod);
	//fscanf(input_module, "NumIOs %d\n", &numIO);


	for (int i = 0; i < numMod; i++) {
		module m;
		string dummy;	
		string str_width, str_height;

		//fscanf(input_module, "%s hardrectilinear (%d, %d) (%lf, %lf) ", m.name, &dummy, &dummy, &m.width, &m.height);
		input >> m.name >> dummy >> dummy >> dummy >> str_width >> str_height;

		m.width = (double)stoi(str_width.substr(1, str_width.size() - 2));
		m.height = (double)stoi(str_height.substr(0, str_height.size() - 1));
		m.moveto(0, 0);
		cout << m.name<<" ("<< m.width<<", "<< m.height<<")"<< endl;
		modlist.push_back(m);
	}

	for (int i = 0; i < numIO; i++) {
		module m;
		string dummy;
		string str_width, str_height;

		//fscanf(input_module, "%s hardrectilinear (%d, %d) (%lf, %lf) ", m.name, &dummy, &dummy, &m.width, &m.height);
		input >> m.name >> dummy >> dummy >> dummy >> str_width >> str_height;

		m.width = (double)stoi(str_width.substr(1, str_width.size() - 2));
		m.height = (double)stoi(str_height.substr(0, str_height.size() - 1));
		m.moveto(0, 0);
		cout << m.name << " (" << m.width << ", " << m.height << ")" << endl;
	
		IOlist.push_back(m);
	}

	return;
}

void parse_net(string file_path_net) {
	ifstream input;
	input.open(file_path_net);
	if (!input.is_open()) {
		cout << "Failed to load *.net" << endl;
	}

	int dummy_int;
	string dummy_string;
	input >> dummy_string >>numNet;
	input >> dummy_string >> dummy_int;
	

	string token;
	while (input >> token) {
		int tmp_name = -1;
		while (input >> token) {
			if (token == "NET") continue;
			if (token == "{") break;
			//string temp_token = token.substr(1, 'end');
			////cout << "hello "<< temp_token << endl;
			//tmp_name = stoi(temp_token);
			const char *cstr = token.c_str();
			net n(cstr);
			//netlist.push_back(n);
			////cout << "hello2 " << tmp_name << endl;
			netlist.push_back(n);
		}

		while (input >> token) {
			//cout << "hello " << token << endl;
			if (token == "}") break;

			if (token.substr(0, 2) == "u_") {
				//string tmp_str = token.substr(2, 'end');
				//int sbname = stoi(tmp_str);
				module* modptr;
				
				for (size_t i = 0; i < modlist.size(); i++) {
					if (modlist[i].name == token) modptr = &(modlist[i]);
				}
				netlist.back().sb.push_back(modptr);
			}
			else { //IO
				//cout << "Wrong file format! token is: " << token << endl; }
				module* IOptr;

				for (size_t i = 0; i < IOlist.size(); i++) {
					if (IOlist[i].name == token) IOptr = &(IOlist[i]);
				}

				netlist.back().IO.push_back(IOptr);
			}
		}

	}
	//cout << dummy_string << " " << numNet << endl;

	return;

}
//



//calculating essential parameters
void calc_C(double* &C) {

	return;
}

void initialize_array(double* &array) {


	return;
}

double cal_edge_weight(net N) {
	int numPin = N.get_pinNum();
	double a = 1.0;
	return a / numPin;
}

//create array with 2*n elements of 0
void create_1Dvector(vector<double> &a, int num) {
	a.resize(2 * num);
	return;
}

//create Matrix with 2*n elements of 0
void create_2Dvector(vector<vector<double>> &a, int num) {
	a.resize(2 * num);
	for (int i = 0; i < 2 * num; i++) {
		a[i].resize(2 * num);
	}
	return;
}

void init_p() {
	//p.resize(2 * totalCell);
	create_1Dvector(p, totalCell);

	int count_mod = numMod;
	int count_IO = numIO;
	for (auto i=0, j= totalCell; i < totalCell; i++, j++) {

		if (count_mod > 0) {
			p[i] = modlist[i].x_cent;
			p[j] = modlist[i].y_cent;
			modlist[i].idx_i = i;
			modlist[i].idx_j = j;
			count_mod--;
		}
		else if (count_IO > 0) {
			int k = i - numMod; //k index for IO, start from 0
			p[i] = IOlist[k].x_cent;
			p[j] = IOlist[k].y_cent;
			IOlist[k].idx_i = i;
			IOlist[k].idx_j = j;
			count_IO--;
		}

		else { cout << "Wrong number of Modules or IOs" << endl; }
	}

	//test print
	cout << "p: " << p.size() << " x 1" << endl;
	for (size_t i = 0; i < p.size(); i++) {
		cout << setw(4) << p[i] ;
	}
	cout << endl;
	cout << endl;
	//
}

void init_d() {
	//d.resize(2 * totalCell);
	create_1Dvector(d, totalCell);
	//int count_mod = numMod;
	//int count_IO = numIO;
	//for (auto i = 0, j = totalCell; i < totalCell; i++, j++) {

	//	if (count_mod > 0) {
	//		p[i] = modlist[i].x_cent;
	//		p[j] = modlist[i].y_cent;
	//		modlist[i].idx_i = i;
	//		modlist[i].idx_j = j;
	//		count_mod--;
	//	}
	//	else if (count_IO > 0) {
	//		int k = i - numMod; //k index for IO, start from 0
	//		p[i] = IOlist[k].x_cent;
	//		p[j] = IOlist[k].y_cent;
	//		IOlist[k].idx_i = i;
	//		IOlist[k].idx_j = j;
	//		count_IO--;
	//	}

	//	else { cout << "Wrong number of Modules or IOs" << endl; }
	//}

	//test print
	cout << "d: " << d.size() << " x 1" << endl;
	for (size_t i = 0; i < d.size(); i++) {
		cout <<setw(4)<<  d[i] ;
	}
	cout << endl;
	cout << endl;
	//
}

void update_C() {
	vector<double> cc;
	create_1Dvector(cc, totalCell);
	C.assign(C.size(), cc);

	for (size_t i = 0; i < netlist.size(); i++) { //for each net

		net temp_net = netlist[i];
		vector<module> temp_modulelist;

		int count_mod = temp_net.sb.size();
		int count_IO = temp_net.IO.size();
		int idx = 0;
		double omega = cal_edge_weight(temp_net); //edge weight of this net
		//cout << "omega: "<< omega << endl;

		//push sb&IO of this net to temp_modulelist
		while (count_mod--) {
			module M = *(temp_net.sb[idx]);
			temp_modulelist.push_back(M);
			idx++;
		}
		idx = 0;
		while (count_IO--) {
			module IO = *(temp_net.IO[idx]);
			temp_modulelist.push_back(IO);
			idx++;
		}
		//
		for (size_t i = 0; i < temp_modulelist.size(); i++) {
			for (size_t j = 0; j < temp_modulelist.size(); j++) {
				if (j == i) continue;
				module m1 = temp_modulelist[i];
				module m2 = temp_modulelist[j];
				//omega12= cal_edge_weight()

				int idx_x1 = m1.idx_i;
				int idx_y1 = m1.idx_j;
				int idx_x2 = m2.idx_i;
				int idx_y2 = m2.idx_j;
				C[idx_x1][idx_x1] = C[idx_x1][idx_x1] + 1 * omega;//1st term of xi...diagonal
				C[idx_x2][idx_x2] = C[idx_x2][idx_x2] + 1 * omega;//3rd term of xj
				C[idx_y1][idx_y1] = C[idx_y1][idx_y1] + 1 * omega;//1st term of yi
				C[idx_y2][idx_y2] = C[idx_y2][idx_y2] + 1 * omega;//3rd term of yj

				C[idx_x1][idx_x2] = C[idx_x1][idx_x2] - 2 * omega;//2nd term of x
				C[idx_x2][idx_x1] = C[idx_x2][idx_x1] - 2 * omega;//2nd term of x...transpose position				
				C[idx_y1][idx_y2] = C[idx_y1][idx_y2] - 2 * omega;//2nd term of y
				C[idx_y2][idx_y1] = C[idx_y2][idx_y1] - 2 * omega;//2nd term of y...transpose position

				//C[idx_x1][idx_x1] = 1 * omega;//1st term of xi...diagonal
				//C[idx_x2][idx_x2] = 1 * omega;//3rd term of xj
				//C[idx_y1][idx_y1] = 1 * omega;//1st term of yi
				//C[idx_y2][idx_y2] = 1 * omega;//3rd term of yj

				//C[idx_x1][idx_x2] = -2 * omega;//2nd term of x
				//C[idx_x2][idx_x1] = -2 * omega;//2nd term of x...transpose position				
				//C[idx_y1][idx_y2] = -2 * omega;//2nd term of y
				//C[idx_y2][idx_y1] = -2 * omega;//2nd term of y...transpose position

			}
		}

	}




	//test print
	cout << "C: " << C.size() << " x " << C[0].size() << endl;
	for (size_t i = 0; i < C.size(); i++) {
		for (size_t j = 0; j < C[i].size(); j++)
		{
			cout << std::setw(4) << C[i][j];
		}
		cout << endl;

	}
	cout << endl;

	return;

}

void init_C() {


	//C.resize(2 * totalCell);

	//for (size_t i = 0; i < C.size(); ++i)
	//{
	//	C[i].resize(2 * totalCell);
	//}

	create_2Dvector(C, totalCell);
	update_C();
	//for (size_t i = 0; i < netlist.size(); i++) { //for each net
	//
	//	net temp_net = netlist[i];
	//	vector<module> temp_modulelist;

	//	int count_mod = temp_net.sb.size();
	//	int count_IO = temp_net.IO.size();
	//	int idx = 0;
	//	double omega = cal_edge_weight(temp_net); //edge weight of this net
	//	//cout << "omega: "<< omega << endl;

	//	//push sb&IO of this net to temp_modulelist
	//	while (count_mod--) {
	//		temp_modulelist.push_back(modlist[temp_net.sb[idx]]);
	//		idx++;
	//	}
	//	idx = 0;
	//	while (count_IO--) {
	//		temp_modulelist.push_back(IOlist[temp_net.IO[idx]]);
	//		idx++;
	//	}
	//	//
	//	for (size_t i = 0; i < temp_modulelist.size(); i++) {
	//		for (size_t j = 0; j < temp_modulelist.size(); j++) {
	//			if (j == i) continue;
	//			module m1 = temp_modulelist[i];
	//			module m2 = temp_modulelist[j];
	//			//omega12= cal_edge_weight()

	//			int idx_x1 = m1.idx_i;
	//			int idx_y1 = m1.idx_j;
	//			int idx_x2 = m2.idx_i;
	//			int idx_y2 = m2.idx_j;

	//			C[idx_x1][idx_x1] = C[idx_x1][idx_x1] + 1*omega;//1st term of xi...diagonal
	//			C[idx_x2][idx_x2] = C[idx_x2][idx_x2] + 1*omega;//3rd term of xj
	//			C[idx_y1][idx_y1] = C[idx_y1][idx_y1] + 1*omega;//1st term of yi
	//			C[idx_y2][idx_y2] = C[idx_y2][idx_y2] + 1*omega;//3rd term of yj

	//			C[idx_x1][idx_x2] = C[idx_x1][idx_x2] - 2 * omega;//2nd term of x
	//			C[idx_x2][idx_x1] = C[idx_x2][idx_x1] - 2 * omega;//2nd term of x...transpose position				
	//			C[idx_y1][idx_y2] = C[idx_y1][idx_y2] - 2 * omega;//2nd term of y
	//			C[idx_y2][idx_y1] = C[idx_y2][idx_y1] - 2 * omega;//2nd term of y...transpose position

	//		}
	//	}

	//}

	//


	////test print
	//cout << "C: " << C.size() << " x " << C[0].size() << endl;
	//for (size_t i = 0; i < C.size(); i++) {
	//	for (size_t j = 0; j < C[i].size(); j++)
	//	{
	//		cout << std::setw(4)<< C[i][j] ;
	//	}
	//	cout << endl;

	//}
	//cout << endl;
	//
}


double cal_ai(module m, double x, double y) { //ai(x, y) = R(x-xi/wi)*R(y-yi/hi)

	double R1, R2;
	double z1, z2;
	
	z1 = (x - m.x_cent) / m.width; //R(x-xi/wi)
	if (z1 > -0.5 && z1 < 0.5) R1 = 1.0;
	else R1 = 0.0;

	z2 = (y - m.y_cent) / m.height; //R(y-yi/hi)
	if (z2 > -0.5 && z2 < 0.5) R2 = 1.0;
	else R2 = 0.0;

	//cout << 

	return R1 * R2;
}

double cal_s() {

	double sum = 0.0; //Sigma( wi*hi)
	for (size_t i = 0; i < modlist.size(); i++) {
		sum = sum + modlist[i].width * modlist[i].height;
	}

	for (size_t i = 0; i < IOlist.size(); i++) {
		sum = sum + IOlist[i].width * IOlist[i].height;
	}

	sum = sum / (grid_h*grid_v); //Sigma(wi*hi)/W*H
	return sum;
}

double cal_A(double x, double y) {

	double R1, R2;
	double z1, z2;

	z1 = x / grid_h; // R(x/W)
	if (z1 > -0.5 && z1 < 0.5) R1 = 1.0;
	else R1 = 0.0;

	z2 = y / grid_v; // R(y/H)
	if (z2 > -0.5 && z2 < 0.5) R2 = 1.0;
	else R2 = 0.0;

	return R1 * R2;

}

double cal_D(double x, double y) {
	double D=0.0;
	double sigma_ai = 0.0;
	double s, A;

	//sigma_ai(x, y)
	for (size_t i = 0; i < modlist.size(); i++) {
		sigma_ai = sigma_ai + cal_ai(modlist[i],x, y);
	}

	for (size_t i = 0; i < IOlist.size(); i++) {
		sigma_ai = sigma_ai + cal_ai(IOlist[i],x, y);
	}

	//s
	s = cal_s();

	//A(x, y,)
	A = cal_A(x, y);

	//	D = sigma_ai(x,y) - s * A(x,y);
	D = sigma_ai - s * A;


	return D;
}

void print_densitymap() {

	density_map.resize(grid_h);
	for (size_t i = 0; i < density_map.size(); i++) {
		density_map[i].resize(grid_v);
	}

	cout << "density map: " << grid_h+1 << " x " << grid_v+1 << endl;
	for (int i = grid_v /2 ; i >= -grid_v/2 ; i--) {
		for (int j = -grid_h/2; j < grid_h /2+1; j++) {
			cout << setw(8) << cal_D(i, j) << "(" << i << "," << j << ")";
		}
		cout << endl;
		cout << endl;
	}

	return;
}

//tools for cal integral
//template<typename value_type, typename function_type>
//inline value_type integral(const value_type a,
//	const value_type b,
//	const value_type tol,
//	function_type func)
//{
//	unsigned n = 1U;
//
//	value_type h = (b - a);
//	value_type I = (func(a) + func(b)) * (h / 2);
//
//	for (unsigned k = 0U; k < 8U; k++)
//	{
//		h /= 2;
//
//		value_type sum(0);
//		for (unsigned j = 1U; j <= n; j++)
//		{
//			sum += func(a + (value_type((j * 2) - 1) * h));
//		}
//
//		const value_type I0 = I;
//		I = (I / 2) + (h * sum);
//
//		const value_type ratio = I0 / I;
//		const value_type delta = ratio - 1;
//		const value_type delta_abs = ((delta < 0) ? -delta : delta);
//
//		if ((k > 1U) && (delta_abs < tol))
//		{
//			break;
//		}
//
//		n *= 2U;
//	}
//
//	return I;
//}
//
//
//template<typename value_type>
//class func_inte_x_0 {
//public:
//
//	func_inte_x_0(const value_type X, const value_type Y,
//		const value_type Y_) : x(X), y(Y), y_(Y_) {
//	};
//
//	value_type  operator()(const value_type  &x_) {
//		return (cal_D(x_, y_) / ((x - x_)*(x - x_) + (y - y_)*(y - y_)))*(x - x_);
//	}
//
//private:
//
//	const double x;
//	const double y;
//	const double y_;
//
//};
//
//template<typename value_type>
//class func_inte_x_1 {
//public:
//
//	func_inte_x_1(const value_type X, const value_type Y,
//		const value_type Y_) : x(X), y(Y), y_(Y_) {
//	};
//
//	value_type  operator()(const value_type  &x_) {
//		return (cal_D(x_, y_) / ((x - x_)*(x - x_) + (y - y_)*(y - y_)))*(y - y_);
//	}
//
//private:
//
//	const double x;
//	const double y;
//	const double y_;
//
//};
//
//template<typename value_type>
//class func_inte_y_0 {
//public:
//	func_inte_y_0(const value_type  A, const value_type  B,
//		const value_type TOL,
//		const value_type  X, const value_type  Y) :
//		x(X), y(Y), tol(TOL), a(A), b(B) {}
//
//	value_type  operator()(const value_type  &y_) {
//		return integral(a, b, tol, func_inte_x_0<double>(x, y, y_));
//	}
//
//private:
//
//	const double x;
//	const double y;
//	const double tol;
//	const double a;
//	const double b;
//
//};
//
//template<typename value_type>
//class func_inte_y_1 {
//public:
//	func_inte_y_1(const value_type  A, const value_type  B,
//		const value_type TOL,
//		const value_type  X, const value_type  Y) :
//		x(X), y(Y), tol(TOL), a(A), b(B) {}
//
//	value_type  operator()(const value_type  &y_) {
//		return integral(a, b, tol, func_inte_x_1<double>(x, y, y_));
//	}
//
//private:
//
//	const double x;
//	const double y;
//	const double tol;
//	const double a;
//	const double b;
//
//};

//double doubleintegral_x(double X, double Y, double limit) {
//	int n = 10e3;//or a large number, the discretization step
//	double dxy = 2 * limit / n;
//	double integral = 0.0;
//	int count = 0;
//	for (int yi = 0; yi < n / 2; yi++) {
//		double y = yi * dxy; //the y-value
//		double y_ = -y; //the y-value
//		for (int xi = 0; xi < n / 2; xi++) {
//			double x = xi * dxy; //the x-value
//			double x_ = -x; //the x-value
//			double D = cal_D(X, Y);
//			//if (!(count++ % 1000)) cout << D << endl;
//			
//			if(!(X==x||X==x_||Y==y||Y==y_))
//			integral += (D / ((X - x)*(X - x) + (Y - y)*(Y - y))*(X - x)) + (D / ((X - x_)*(X - x_) + (Y - y_)*(Y - y_))*(X - x_)); //function call
//			//cout <<count++ << " "<< integral << endl;
//		}
//	}
//
//	return integral;
//}
//
//double doubleintegral_y(double X, double Y, double limit) {
//	int n = 10e3;//or a large number, the discretization step
//	double dxy = 2 * limit / n;
//	double integral = 0.0;
//	int count = 0;
//	for (int yi = 0; yi < n / 2; yi++) {
//		double y = yi * dxy; //the y-value
//		double y_ = -y;//the y-value
//		for (int xi = 0; xi < n / 2; xi++) {
//			double x = xi * dxy; //the x-value
//			double x_ = -x; //the x-value
//			double D = cal_D(X, Y);
//			//cout << "hello2" << endl;
//			if (!(X == x || X == x_ || Y == y || Y == y_))
//			integral += (D / ((X - x)*(X - x) + (Y - y)*(Y - y))*(Y - y)) + (D / ((X - x_)*(X - x_) + (Y - y_)*(Y - y_))*(Y - y_)); //function call
//			
//			//if ( !(count++ %10000000) ) cout << integral << endl;
//			//cout << integral << endl;
//		}
//	}
//
//	return integral;
//}

pair<double, double> doubleintegral(double X, double Y, double limit) {
	int n = Segment;//or a large number, the discretization step
	double dxy = 2 * limit / n;
	double integral_0 = 0.0;
	double integral_1 = 0.0;
	int count = 0;
	for (int yi = 0; yi < n / 2; yi++) {
		double y = yi * dxy; //the y-value
		double y_ = -y;//the y-value
		for (int xi = 0; xi < n / 2; xi++) {
			double x = xi * dxy; //the x-value
			double x_ = -x; //the x-value
			double D_plus = cal_D(x, y);
			double D_minus = cal_D(x_, y_);
			//cout << "hello2" << endl;
			if (!(X == x || X == x_ || Y == y || Y == y_)) {
				double recip_eucD_D_plus = D_plus / ((X - x)*(X - x) + (Y - y)*(Y - y)); //  recip_eucD_D = D/|r-r'|^2
				double recip_eucD_D_minus = D_minus / ((X - x_)*(X - x_) + (Y - y_)*(Y - y_));
				integral_0 += (recip_eucD_D_plus*(X - x)) + (recip_eucD_D_minus*(X - x_)); //function call
				integral_1 += (recip_eucD_D_plus*(Y - y)) + (recip_eucD_D_minus*(Y - y_)); //function call
			}
				//if ( !(count++ %10000000) ) cout << integral << endl;
				//cout << integral << endl;
		}
	}
	coef_k = 1.0;
	pair<double, double> integral_pair= make_pair(integral_0*coef_k/(2*M_PI), integral_1*coef_k / (2 * M_PI));

	return integral_pair;
}

//double f_no_k_0(const double x, const double y) {
//	double limit_x = Inte_distance * grid_h;
//	//double limit_y = Inte_distance * grid_v;
//	//const double tol = tolerance;
//
//
//	return doubleintegral_x(x, y, limit_x) / (2 * M_PI);
//}
//
//double f_no_k_1(const double x, const double y) {
//	double limit_x = Inte_distance * grid_h;
//	//double limit_y = Inte_distance * grid_v;
//	//const double tol = tolerance;
//
//
//	return doubleintegral_y(x, y, limit_x) / (2 * M_PI);
//}

/*double f_no_k_0(const double x, const double y) {
	double limit_x = Inte_distance * grid_h;
	double limit_y = Inte_distance * grid_v;
	const double tol = tolerance;


	return integral(-limit_y, limit_y, tol, func_inte_y_0<double>(-limit_x, limit_x, tol, x, y)) / (2 * M_PI);
}

double f_no_k_1(const double x, const double y) {
	double limit_x = Inte_distance * grid_h;
	double limit_y = Inte_distance * grid_v;
	const double tol = tolerance;


	return integral(-limit_y, limit_y, tol, func_inte_y_1<double>(-limit_x, limit_x, tol, x, y)) / (2 * M_PI);
}*/
//

void init_e() {
	create_1Dvector(e, totalCell);
	//for (size_t i = 0; i < modlist.size(); i++) {
	//	module m = modlist[i];
	//	e[m.idx_i] = f_no_k_0(m.x_cent, m.y_cent);
	//	e[m.idx_j] = f_no_k_1(m.x_cent, m.y_cent);

	//	cout << "e[i, n+i] : ("<<e[m.idx_i] << ", " << e[m.idx_j] <<" )"<< endl;
	//}

	//for (size_t i = 0; i < IOlist.size(); i++) {
	//	module m = IOlist[i];
	//	e[m.idx_i] = f_no_k_0(m.x_cent, m.y_cent);
	//	e[m.idx_j] = f_no_k_1(m.x_cent, m.y_cent);

	//	cout << "e[i, n+i] : (" << e[m.idx_i] << ", " << e[m.idx_j] << " )" << endl;
	//}


	//test print
	cout << "e: " << e.size() << " x 1" << endl;
	for (size_t i = 0; i < e.size(); i++) {
		cout << e[i] << " ";
	}
cout << endl;
cout << endl;
//

}

void update_e() {

	for (size_t i = 0; i < modlist.size(); i++) {
		module m = modlist[i];
		//double limit = Inte_distance * (grid_h+grid_v)/2;
		double limit = max(grid_h, grid_v);
		pair<double, double> f = doubleintegral(m.x_cent, m.y_cent, limit);
		/*		e[m.idx_i] = f_no_k_0(m.x_cent, m.y_cent);
				e[m.idx_j] = f_no_k_1(m.x_cent, m.y_cent);		*/
		e[m.idx_i] = f.first;
		e[m.idx_j] = f.second;


		cout << "e[i, n+i] : (" << e[m.idx_i] << ", " << e[m.idx_j] << " )" << endl;
	}

	for (size_t i = 0; i < IOlist.size(); i++) {
		module m = IOlist[i];
		double limit = Inte_distance * (grid_h + grid_v) / 2;
		pair<double, double> f = doubleintegral(m.x_cent, m.y_cent, limit);
		/*		e[m.idx_i] = f_no_k_0(m.x_cent, m.y_cent);
				e[m.idx_j] = f_no_k_1(m.x_cent, m.y_cent);		*/
		e[m.idx_i] = f.first;
		e[m.idx_j] = f.second;

		cout << "e[i, n+i] : (" << e[m.idx_i] << ", " << e[m.idx_j] << " )" << endl;
	}


	//test print
	cout << "updated e: " << e.size() << " x 1" << endl;
	for (size_t i = 0; i < e.size(); i++) {
		cout << e[i] << " ";
	}
	cout << endl;
	cout << endl;
	//
}

void cal_cost_func() {

	for (size_t i = 0; i < C.size(); i++) {
		for (size_t j = 0; j < C[0].size(); j++) {
			cost[i] = cost[i] + C[i][j] * p[i];
		}
		cost[i] = cost[i] + d[i] + e[i];
	}
	//test print
	cout << "cost: " << cost.size() << " x 1" << endl;
	for (size_t i = 0; i < cost.size(); i++) {
		cout << setw(5) << cost[i] << endl;
	}
	cout << endl;
	cout << endl;
	//

	return;
}

vector<double> solve_p() {

	vector<double> B;
	B.resize(2 * totalCell, 0.0);
	for (size_t i = 0; i < B.size(); i++) {
		B[i] = -e[i];
	}

	//vector<double> P = conjugateGradientSolver(C , B);
	vector<double> P = conjugate_gradient(C, B, step_size);

	cout << "Solves CP = B\n"; // B=-e
	print("\nA:", C);
	print("\nB:", B);
	print("\nX:", P);
	print("\nCheck AX:", matrixTimesVector(C, P));

	return P;
}



void IO_boundary_force() {

	cout << "IO_boundary_force added: " << endl;
	for (size_t i = 0; i < IOlist.size(); i++) {
		module m = IOlist[i];
		double addi_force_x = abs(e[m.idx_i])*boundary_force;
		double addi_force_y = abs(e[m.idx_j])*boundary_force;
		double limit_w = grid_h / 2.0;
		double limit_h = grid_v / 2.0;
		double epsilon = 0.75;




		if (((abs(abs(m.x_cent) - limit_w)) <= epsilon) && (abs(abs(m.y_cent) - limit_h) > epsilon))  //X OK Y NOT
		e[m.idx_i] = 0;

		else if (((abs(abs(m.x_cent) - limit_w)) > epsilon) && ((abs(abs(m.y_cent) - limit_h)) <= epsilon))  //X NOT Y OK
		e[m.idx_j] = 0;

		else if (((abs(abs(m.x_cent) - limit_w)) <= epsilon) && ((abs(abs(m.y_cent) - limit_h)) <= epsilon)) //X OK Y OK >>>AT CORNER, NOT GOOD
		{
		e[m.idx_i] *= 30;
		e[m.idx_j] *= 30; }

		else if (((abs(abs(m.x_cent) - limit_w)) <= epsilon) && ((abs(abs(m.y_cent) - limit_h)) <= epsilon)){  //X NOT Y NOT

			if (abs(m.x_cent) >= abs(m.y_cent)) {
				e[m.idx_i] += m.x_cent >= 0 ? addi_force_x : -addi_force_x;
				double new_x = m.x_cent >= 0 ? ((grid_h / 2.0) - 0.5) : ((-grid_h / 2.0) + 0.5);
				IOlist[i].moveto(new_x, IOlist[i].y_cent);
			}
			else if (abs(m.x_cent) < abs(m.y_cent)) {
				e[m.idx_j] += m.x_cent >= 0 ? addi_force_x : -addi_force_x;
				double new_y = m.y_cent >= 0 ? ((grid_v / 2.0) - 0.5) : ((-grid_v / 2.0) + 0.5);
				IOlist[i].moveto(IOlist[i].x_cent, new_y);
			}

		}

		cout << "e[i, n+i] : (" << e[m.idx_i] << ", " << e[m.idx_j] << " )" << endl;
	}

	//test print
	cout << "updated e: " << e.size() << " x 1" << endl;
	for (size_t i = 0; i < e.size(); i++) {
		cout << e[i] << " ";
	}
	cout << endl;
	cout << endl;
	//
}

void move_IOs_to_boundary() {

	for (size_t i = 0; i < IOlist.size(); i++) {

		if (abs(IOlist[i].x_cent) >= abs(IOlist[i].y_cent)) {
			double new_x = IOlist[i].x_cent >= 0 ? ((grid_h / 2.0) - 0.5) : ((-grid_h / 2.0) + 0.5);
			IOlist[i].moveto(new_x, IOlist[i].y_cent);
		}
		else if (abs(IOlist[i].x_cent) < abs(IOlist[i].y_cent)) {
			double new_y = IOlist[i].y_cent >= 0 ? ((grid_v / 2.0) - 0.5) : ((-grid_v / 2.0) + 0.5);
			IOlist[i].moveto(IOlist[i].x_cent, new_y);
		}

	}

}

void move_IO_to_corner() {
	for (double i = 0.0; i < (double)IOlist.size(); i++) {
		
		IOlist[i].moveto((-grid_h / 2) + IOlist[i].width/2 + i*2, (-grid_v / 2) + IOlist[i].height/2);
		
		//cout << -(grid_h / 2 + IOlist[i].width / 2) << " " << (-grid_v / 2) + IOlist[i].height / 2 << endl;
		
		cout << "helo "<< IOlist[i].x_cent<<" "<< IOlist[i].y_cent << endl;
	}

	for (size_t i = 0; i < modlist.size(); i++) {
		cout << "sb[" << i << "].name= " << modlist[i].name << " w= " << modlist[i].width;
		cout << " h= " << modlist[i].height << " x= " << modlist[i].x << " y= " << modlist[i].y;
		cout << " x_cent= " << modlist[i].x_cent << " y_cent= " << modlist[i].y_cent << endl;
	}
	cout << "IO.size= " << IOlist.size() << endl;
	for (size_t i = 0; i < IOlist.size(); i++) {
		cout << "IO[" << i << "].name= " << IOlist[i].name;
		cout << " w= " << IOlist[i].width << " h= " << IOlist[i].height << " x= " << IOlist[i].x << " y= " << IOlist[i].y;
		cout << " x_cent= " << IOlist[i].x_cent << " y_cent= " << IOlist[i].y_cent << endl;
	}



}
//get matrix inverse
double getDeterminant(const std::vector<std::vector<double>> vect) {
	if (vect.size() != vect[0].size()) {
		throw std::runtime_error("Matrix is not quadratic");
	}
	int dimension = vect.size();

	if (dimension == 0) {
		return 1;
	}

	if (dimension == 1) {
		return vect[0][0];
	}

	//Formula for 2x2-matrix
	if (dimension == 2) {
		return vect[0][0] * vect[1][1] - vect[0][1] * vect[1][0];
	}

	double result = 0;
	int sign = 1;
	for (int i = 0; i < dimension; i++) {

		//Submatrix
		std::vector<std::vector<double>> subVect(dimension - 1, std::vector<double>(dimension - 1));
		for (int m = 1; m < dimension; m++) {
			int z = 0;
			for (int n = 0; n < dimension; n++) {
				if (n != i) {
					subVect[m - 1][z] = vect[m][n];
					z++;
				}
			}
		}

		//recursive call
		result = result + sign * vect[0][i] * getDeterminant(subVect);
		sign = -sign;
	}

	return result;
}

std::vector<std::vector<double>> getTranspose(const std::vector<std::vector<double>> matrix1) {

	//Transpose-matrix: height = width(matrix), width = height(matrix)
	std::vector<std::vector<double>> solution(matrix1[0].size(), std::vector<double>(matrix1.size()));

	//Filling solution-matrix
	for (size_t i = 0; i < matrix1.size(); i++) {
		for (size_t j = 0; j < matrix1[0].size(); j++) {
			solution[j][i] = matrix1[i][j];
		}
	}
	return solution;
}

std::vector<std::vector<double>> getCofactor(const std::vector<std::vector<double>> vect) {
	if (vect.size() != vect[0].size()) {
		throw std::runtime_error("Matrix is not quadratic");
	}

	std::vector<std::vector<double>> solution(vect.size(), std::vector<double>(vect.size()));
	std::vector<std::vector<double>> subVect(vect.size() - 1, std::vector<double>(vect.size() - 1));

	for (std::size_t i = 0; i < vect.size(); i++) {
		for (std::size_t j = 0; j < vect[0].size(); j++) {

			int p = 0;
			for (size_t x = 0; x < vect.size(); x++) {
				if (x == i) {
					continue;
				}
				int q = 0;

				for (size_t y = 0; y < vect.size(); y++) {
					if (y == j) {
						continue;
					}

					subVect[p][q] = vect[x][y];
					q++;
				}
				p++;
			}
			solution[i][j] = pow(-1, i + j) * getDeterminant(subVect);
		}
	}
	return solution;
}

std::vector<std::vector<double>> getInverse(const std::vector<std::vector<double>> vect) {
	if (getDeterminant(vect) == 0) {
		throw std::runtime_error("Determinant is 0");
	}

	double d = 1.0 / getDeterminant(vect);
	std::vector<std::vector<double>> solution(vect.size(), std::vector<double>(vect.size()));

	for (size_t i = 0; i < vect.size(); i++) {
		for (size_t j = 0; j < vect.size(); j++) {
			solution[i][j] = vect[i][j];
		}
	}

	solution = getTranspose(getCofactor(solution));

	for (size_t i = 0; i < vect.size(); i++) {
		for (size_t j = 0; j < vect.size(); j++) {
			solution[i][j] *= d;
		}
	}

	return solution;
}

void printMatrix(const std::vector<std::vector<double>> vect) {
	for (std::size_t i = 0; i < vect.size(); i++) {
		for (std::size_t j = 0; j < vect[0].size(); j++) {
			std::cout << std::setw(8) << vect[i][j] << " ";
		}
		std::cout << "\n";
	}
}


//vector<double> solve_p_byInverse() {
void solve_p_byInverse() {

	vector<double> B;
	B.resize(2 * totalCell, 0.0);
	for (size_t i = 0; i < B.size(); i++) {
		B[i] = -e[i];
	}
	printMatrix(getInverse(C));
	//vector<double> P = conjugateGradientSolver(C, B);

	//cout << "Solves CP = B\n"; // B=-e
	//print("\nA:", C);
	//print("\nB:", B);
	//print("\nX:", P);
	//print("\nCheck AX:", matrixTimesVector(C, P));

	//return P;
}
//

void normalize_e() {

	vector<double> copy_e = e;
	cout << "copy_e"<<endl;
	for (size_t j = 0; j < copy_e.size(); j++) {
		cout << copy_e[j] << " ";
	}
	cout << endl;
	sort(copy_e.begin(), copy_e.end());

	cout << "sorted copy_e" << endl;
	for (size_t j = 0; j < copy_e.size(); j++) {
		cout << copy_e[j] << " ";
	}
	cout << endl;

	double largest_abs = abs(copy_e.front()) > abs(copy_e.back()) ? abs(copy_e.front()) : abs(copy_e.back());
	/*double largest_abs = *max_element(e.begin(), e.end(), [](const double& a, const double& b)
	{
		return abs(a) < abs(b);
	});*/

	//largest_abs = abs(largest_abs);
	//double it = *max_element(e.begin(), e.end()); // c++11
	cout << "Max in e: " << largest_abs << endl;

	for (size_t i = 0; i < modlist.size(); i++) {
		e[modlist[i].idx_i] = e[modlist[i].idx_i] /largest_abs * normal_factor_mod;
		e[modlist[i].idx_j] = e[modlist[i].idx_j] /largest_abs * normal_factor_mod;
	}

	for (size_t i = 0; i < IOlist.size(); i++) {
		e[IOlist[i].idx_i] = e[IOlist[i].idx_i] / largest_abs * normal_factor_IO;
		e[IOlist[i].idx_j] = e[IOlist[i].idx_j] / largest_abs * normal_factor_IO;
	}


	cout << "nomalized e: " << e.size() << " x 1" << endl;
	for (size_t i = 0; i < e.size(); i++) {
		cout << e[i] << " ";
	}
	cout << endl;
}

void update_module_coord(vector<double> p) {

	for (size_t i = 0; i < modlist.size(); i++) {
		int idx_x = modlist[i].idx_i;
		int idx_y = modlist[i].idx_j;
		double limit_w = grid_h / 2.0;
		double limit_h = grid_v / 2.0;
		double x = abs(p[idx_x]) > limit_w ? (p[idx_x] > 0.0? limit_h : -limit_w) : p[idx_x];
		double y = abs(p[idx_y]) > limit_h ? (p[idx_y] > 0.0? limit_h : -limit_h) : p[idx_x];
		modlist[i].moveto(p[idx_x], p[idx_y]);
	}

	for (size_t i = 0; i < IOlist.size(); i++) {
		int idx_x = IOlist[i].idx_i;
		int idx_y = IOlist[i].idx_j;
		IOlist[i].moveto(p[idx_x], p[idx_y]);
	}
	move_IOs_to_boundary();

	return;
}

void print_all_mod() {

	for (size_t i = 0; i < modlist.size(); i++) {
		cout << "sb[" << i << "].name= " << modlist[i].name << " w= " << modlist[i].width;
		cout << " h= " << modlist[i].height << " x= " << modlist[i].x << " y= " << modlist[i].y;
		cout << " x_cent= " << modlist[i].x_cent << " y_cent= " << modlist[i].y_cent << endl;
	}
	cout << "IO.size= " << IOlist.size() << endl;
	for (size_t i = 0; i < IOlist.size(); i++) {
		cout << "IO[" << i << "].name= " << IOlist[i].name;
		cout << " w= " << IOlist[i].width << " h= " << IOlist[i].height << " x= " << IOlist[i].x << " y= " << IOlist[i].y;
		cout << " x_cent= " << IOlist[i].x_cent << " y_cent= " << IOlist[i].y_cent << endl;
	}
	return;
}

bool check_IO_position() {
	bool flag = true;
	double limit_w = grid_h / 2.0 -0.5;
	double limit_h = grid_v / 2.0 -0.5;
	double epsilon = 0.5;
	for (size_t i = 0; i < IOlist.size(); i++) {
		if (abs((abs(IOlist[i].x_cent) - limit_w)) > epsilon) flag = false;
		//cout << (abs(IOlist[i].x_cent) - limit_w) << endl;
		if (abs((abs(IOlist[i].y_cent) - limit_h)) > epsilon) flag = false;
	}
	
	cout << "flag: " << flag << endl;
	return flag;
}

void print_net() {
	cout << endl;
	cout << "--[Netlist]--" << endl;
	for (size_t i = 0; i < netlist.size(); i++) {
		cout << netlist[i].name << ": " << endl;
		for (size_t j = 0; j < netlist[i].sb.size(); j++)
			cout << netlist[i].sb[j]->name << " ";
		for (size_t j = 0; j < netlist[i].IO.size(); j++)
			cout << netlist[i].IO[j]->name << " ";
		cout << endl;
	}

}

int main(int argc, char *argv[]) {

	//FILE* input_module = fopen(argv[1], "r");
	//FILE* input_module = fopen("./testcase/case1.module", "r");
	string file_path_module(argv[1]);
	string file_path_net(argv[2]);

	//if (input_module == NULL) {
	//	perror("file read success failed: ");
	//	return 1;
	//}
	//else {
	//	cout << "file read success" << endl;
	//}

	parse_module(file_path_module);
	parse_net(file_path_net);
	//print_net();

	//test print
	cout << "grid : " << grid_h << " " << grid_v << endl;
	cout << "cent grid : " << h_cent << " " << v_cent << endl;
	cout << "sb.size= " << modlist.size() << endl;
	for (size_t i = 0; i < modlist.size(); i++) {
		cout << "sb[" << i << "].name= " << modlist[i].name << " w= " << modlist[i].width;
		cout << " h= " << modlist[i].height << " x= " << modlist[i].x << " y= " << modlist[i].y;
		cout << " x_cent= " << modlist[i].x_cent << " y_cent= " << modlist[i].y_cent << endl;
	}
	cout << "IO.size= " << IOlist.size() << endl;
	for (size_t i = 0; i < IOlist.size(); i++) {
		cout << "IO[" << i << "].name= " << IOlist[i].name;
		cout << " w= " << IOlist[i].width << " h= " << IOlist[i].height << " x= " << IOlist[i].x << " y= " << IOlist[i].y;
		cout << " x_cent= " << IOlist[i].x_cent << " y_cent= " << IOlist[i].y_cent << endl;
	}
	//

	totalCell = numMod + numIO;
	cout << "-------iter 1-------" << endl;
	cout << endl;
	init_p();
	init_d();
	init_C();


	init_e(); //should set to 0
	move_IO_to_corner();
	//cout << " D(-4.5, -4.5)" << cal_D(-4.5, -4.5) << endl;
	//cout << " D(-3.5, -4.5)" << cal_D(-3.5, -4.5) << endl;
	//cout << " D(-2.5, -4.5)" << cal_D(-2.5, -4.5) << endl;
	//cout << " D(-1.5, -4.5)" << cal_D(-1.5, -4.5) << endl;
	//cout << " D(-0.5, -4.5)" << cal_D(-0.5, -4.5) << endl;
	//cout << " D(0.5, -4.5)" << cal_D(0.5, -4.5) << endl;
	//cout << " D(1.5, -4.5)" << cal_D(1.5, -4.5) << endl;
	//cout << " D(2.5, -4.5)" << cal_D(2.5, -4.5) << endl;
	//cout << " D(3.5, -4.5)" << cal_D(3.5, -4.5) << endl;
	//print_densitymap();
	update_e();
	normalize_e();

	create_1Dvector(cost, totalCell);
	
	//IO_boundary_force();

	//Loop1

	update_module_coord(solve_p());

	print_all_mod();

	//loop 2
	cout << "-------iter 2-------" << endl ;
	cout << endl;
	update_C();
	update_e();
	normalize_e();
	//IO_boundary_force();

	update_module_coord(solve_p());
	print_all_mod();
	//solve_p_byInverse();
	/*for(int i = -10*grid_h; i < 10* grid_h ; i++)
		for(int j = -10*grid_h; j < 10* grid_h ; j++)
		
	cout <<i<<" "<<j<<" : " << cal_D(i, j) << endl;*/

	int iter = 0;
	do{
		cout << "-------iter " <<iter++<<"-------" << endl;
		cout << endl;
		update_C();
		update_e();
		normalize_e();
		//IO_boundary_force();
		update_module_coord(solve_p());
		print_all_mod();
	} while (!check_IO_position());


	return 0;
}
