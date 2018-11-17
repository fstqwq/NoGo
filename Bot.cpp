/* This is a simple NoGo bot written by fstqwq(YANG Zonghan).
 * Current Version : Alpha-Beta + UCT Version
 * For my lovely Lily
 */
#include <bits/stdc++.h>
#include "submit.h"
using namespace std;

/****** Magic ******/

#pragma GCC optimize("-O3")
#define __ __attribute__((optimize("-O3")))
#define inline __ __inline __attribute__((__always_inline__, __artificial__))

/***** Constants *****/

extern int ai_side;
string ai_name = "Bot";
typedef signed char stone;
const int N = 9;
const stone NONE = 0, BLACK = 1, WHITE = 2;
const int INF = 999, TLE = 2333;
const int dx[] = {1, -1, 0, 0}, dy[] = {0, 0, 1, -1};
const int x8[] = {-1, -1, 1, 1, 1, -1, 0, 0}, y8[] = {-1, 1, -1, 1, 0, 0, 1, -1};
const time_t TIME_LIMIT = time_t(CLOCKS_PER_SEC * 0.9);
const int Score[5] = {99, 32, 16, 8, 4};
const int DefaultDotLimit = 99; // deprecated

/****** Data Structures and Functions ******/

double sqLOG[300000], sq[300000], EXP[100]; // precalculate anything related to sqrt(), log(), exp()
inline double Rate(const int x) {
	return x > 0 ? 1 - EXP[x] : EXP[-x];
}
inline constexpr stone Switch(const stone x) {return stone(3 - x);}
inline bool legal(const int i, const int j) {return i >= 0 && i < N && j >= 0 && j < N;}
inline int min(int a, int b) {return a < b ? a : b;}
inline int max(int a, int b) {return a > b ? a : b;}

inline uint64_t xorshift128plus() { // xorshift128+, from Wikipedia
	static uint64_t s[2] = {200312llu + uint64_t(time(NULL)) * uint64_t(s), 9999999919260817llu ^ uint64_t(dx)};
	uint64_t x = s[0];
	uint64_t const y = s[1];
	s[0] = y;
	x ^= x << 23; // a
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
	return s[1] + y;
}

inline unsigned Rand() {
	return (unsigned)xorshift128plus();
}

template <class T> inline T Rand(const T st, const T ed) {
	return T(xorshift128plus() % (ed - st + 1)) + st;
}

template <class RandomIt> inline void Shuffle(RandomIt first, RandomIt last) { // random_shuffle()
	for (int i = int(last - first) - 1; i > 0; --i) {
		swap(first[i], first[Rand(0, i)]);
	}
}

template <int SIZE> struct UFset { // Union-Find Set
	int f[SIZE];
	inline UFset () {
		for (int i = 0; i < SIZE; i++) f[i] = i;
	}
	inline void clear () {
		for (int i = 0; i < SIZE; i++) f[i] = i;
	}
	inline int & operator [] (const int x) {return f[x];}
	__ int getf(const int x) {
		return f[x] == x ? x : (f[x] = getf(f[f[x]])); // magic
	}
	inline void merge(const int x, const int y) {
		if (getf(x) != getf(y)) {
			f[getf(y)] = getf(x); // no need of optimization
		}
	}
	inline bool isRoot(const int x) {
		return f[x] == x;
	}
};

struct Point { // pair of int
	int x, y;
	inline Point () {}
	inline Point (const int a, const int b) {x = a, y = b;}
	inline Point (const int a) {x = a / N, y = a % N;}
	inline Point (const pair <int, int> a) {x = a.first, y = a.second;}
	inline Point operator = (const Point b) {x = b.x, y = b.y; return *this;}
	inline bool operator < (const Point &a) const {return x < a.x || (x == a.x && y < a.y);}
	inline operator pair<int, int>() const {return make_pair(x, y);}
};

int lasteval; // Record last value of evaluation, optimization of constant
bool isUCT; // Avoid passing too much bools, optimization of constant

struct Board { // Chessoard
	struct EyeDetect { // A data structure that can evaluate the board by eye detection
		int a[N][N];
		inline int at(const int i, const int j) const {return a[i][j];}
		inline int at(const Point x) const {return a[x.x][x.y];}
		inline void eval(const Board &board, const stone hand) { // Complexity : O(N ^ 2) with constant of 5
			memset(a, 0, sizeof a);
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) if (!board.at(i, j)) {
					int c[3] = {0};
					for (int d = 0; d < 4; d++) {
						int tx = i + dx[d], ty = j + dy[d];
						if (board.at(tx, ty) >= 0) c[board.at(tx, ty)]++;
					}
					if (!c[1] || !c[2]) {
						const int val = Score[c[0]];
						if (c[Switch(hand)]) a[i][j] += val;
						for (int d = 0; d < 4; d++) {
							int tx = i + dx[d], ty = j + dy[d];
							if (!board.at(tx, ty)) {
								a[tx][ty] += val;
							}
						}
					}
					for (int d = 0; d < 4; d++) {
						int tx = i + x8[d], ty = j + y8[d];
						if (board.at(tx, ty) >= 0) c[board.at(tx, ty)]++;
					}
					a[i][j] += c[hand];
				}
			}
			a[0][0] += 8; a[N - 1][N - 1] += 8;
			a[0][N - 1] += 8; a[N - 1][0] += 8;
		}
	};

	stone g[N][N];
	inline Board () {memset(g, 0, sizeof g);}
	inline Board (char x) { memset(g, x, sizeof g); }
	inline stone * operator [] (const int x) { return g[x]; }
	inline stone at(const int i, const int j) const { if (i >= 0 && i < N && j >= 0 && j < N) return g[i][j]; else return -1;}
	inline stone at(const Point x) const {if (x.x >= 0 && x.x < N && x.y >= 0 && x.y < N) return g[x.x][x.y]; else return -1;}
	inline void set(const int i, const int j, const char k) {g[i][j] = k;} 
	inline void set(const Point x, const char k) {g[x.x][x.y] = k;}
	inline void reset(const int i, const int j) {g[i][j] = 0;}
	inline void reset(const Point x) {g[x.x][x.y] = 0;}

	inline bool valid() const { // Complexity : O(N ^ 2) // Unused
		UFset<N * N> f;
		int chi[N * N];
		memset(chi, 0, sizeof chi);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (g[i][j]) {
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj)) {
						if (g[ti][tj]) {
							if (g[ti][tj] == g[i][j]) {
								f.merge(i * N + j, ti * N + tj);
							}
						}
						else chi[i * N + j]++;
					}
				}
			}
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if (g[i][j] && !f.isRoot(i * N + j)) {
					chi[f.getf(i * N + j)] += chi[i * N + j]; 
				}
			}
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if (g[i][j] && f.isRoot(i * N + j) && !chi[i * N + j]) {
					return false;
				}
			}
		}
		return true;
	}

	inline void sort_by_eval(vector <Point> &a, const bool MY[N][N], const bool EN[N][N], const stone hand, const int chi[N * N]) { // Complexity : sort(s ^ 2)
		static pair<int, Point> ret[81];
		static EyeDetect Eval;	
		Eval.eval(*this, hand);
		int c = 0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (MY[i][j]) {
				int value = (EN[i][j] * 16 + chi[i * N + j] * 4 + Eval.at(i, j)) * 1024 + Rand() % (isUCT ? 16384 : 2048); // Consider value of enemy's availability and randomization
				ret[c++] = make_pair(-value, (Point){i, j});
			}
		}
		sort(ret, ret + c);
		a.resize(c); // Avoid push_back()
		for (int i = 0; i < c; i++) a[i] = ret[i].second;
	}

	inline void getValidPoints(const stone hand, vector <Point> &ret) { // Complexity : O(N ^ 2) + sort_by_eval()
		static UFset<N * N> f;
		static int chi[N * N];
		static bool MY[N][N], EN[N][N];

		// Clear
		ret.clear();
		f.clear();
		memset(chi, 0, sizeof chi);
		memset(MY, 0, sizeof MY);
		memset(EN, 0, sizeof EN);

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (g[i][j]) {
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj) && g[ti][tj] && g[ti][tj] == g[i][j]) {
						f.merge(i * N + j, ti * N + tj);
					}
				}
			}
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (!g[i][j]) {
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj) && g[ti][tj] && chi[f.getf(ti * N + tj)] >= 0) {
						chi[f[ti * N + tj]] = -(chi[f[ti * N + tj]] + 1);
					}
				}
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj) && g[ti][tj] && chi[f.getf(ti * N + tj)] < 0) {
						chi[f[ti * N + tj]] = -chi[f[ti * N + tj]];
					}
				}
			}
		}
		int aval = 0, eavl = 0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (!g[i][j]) {
				int o[3] = {0}, c[3] = {0}, e = 0;	
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj)) {
						if (g[ti][tj]) {
							c[g[ti][tj]]++;
							if (chi[f.getf(ti * N + tj)] == 1) o[g[ti][tj]]++;
						}
						else e++;
					}
				}
				if ((o[1] == 0 || o[1] < c[1] || e) && (!o[2]) && (c[1] + e)) (hand == BLACK ? (aval++, MY) : (eavl++, EN))[i][j] = 1;
				if ((o[2] == 0 || o[2] < c[2] || e) && (!o[1]) && (c[2] + e)) (hand == WHITE ? (aval++, MY) : (eavl++, EN))[i][j] = 1;
				chi[i * N + j] = o[hand];
			}
		}
		lasteval = aval - eavl;
		sort_by_eval(ret, MY, EN, hand, chi); // shuffle the return value by eye evaluation
	}

	inline int eval(const stone hand) const { // Complexity : O(N ^ 2)
		int ret = 0;
		static UFset<N * N> f;
		static int chi[N * N];
		f.clear();
		memset(chi, 0, sizeof chi);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (g[i][j]) {
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj) && g[ti][tj] && g[ti][tj] == g[i][j]) {
						f.merge(i * N + j, ti * N + tj);
					}
				}
			}
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (!g[i][j]) {
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj) && g[ti][tj] && chi[f.getf(ti * N + tj)] >= 0) {
						chi[f[ti * N + tj]] = -(chi[f[ti * N + tj]] + 1);
					}
				}
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj) && g[ti][tj] && chi[f.getf(ti * N + tj)] < 0) {
						chi[f[ti * N + tj]] = -chi[f[ti * N + tj]];
					}
				}
			}
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (!g[i][j]) {
				int o[3] = {0}, c[3] = {0}, e = 0;	
				for (int d = 0; d < 4; d++) {
					int ti = i + dx[d], tj = j + dy[d]; 
					if (legal(ti, tj)) {
						if (g[ti][tj]) {
							c[g[ti][tj]]++;
							if (chi[f.getf(ti * N + tj)] == 1) o[g[ti][tj]]++;
						}
						else e++;
					}
				}
				if ((o[1] == 0 || o[1] < c[1] || e) && (!o[2]) && (c[1] + e)) ret++;
				if ((o[2] == 0 || o[2] < c[2] || e) && (!o[1]) && (c[2] + e)) ret--;
			}
		}
		return lasteval = (hand == BLACK ? ret : -ret);
	}
};

/****** Main Strategies ******/

Board Main;
stone my, en;
int Step = 0;
time_t StartTime;

inline bool isTLE() {return clock() - StartTime > TIME_LIMIT;}

class Alpha_Beta {
	private:
	__ int dfs(const int Steps, const stone hand, int alpha, const int beta) {
		if (!Steps) return Main.eval(hand);
		if (isTLE()) return -TLE;
		vector <Point> choice;
		Main.getValidPoints(hand, choice);
		int width = min(16 + (choice.size() < 50) * 16 + (hand == en) * 16, (int)choice.size()); // Limit the width of searching, assuming that enemy is wiser
		for (int j = 0 ; j < width; j++) {
			const Point &i = choice[j];
			Main.set(i, hand);
			int value = -dfs(Steps - 1, Switch(hand), -beta, -alpha);
			Main.reset(i);
			if (value == TLE) return -TLE;
			if (value >= beta) return beta;
			if (value > alpha) alpha = value;
		}
		return alpha;
	}

	int ans_value = -INF;
	Point ans = {0, 0};

	public:
	inline pair<int, int> Action() {
		isUCT = 0;
		vector <Point> choice;
		Main.getValidPoints(my, choice);

		ans = *choice.begin(); // assert choice.size() > 0
		ans_value = -1;

		unsigned StepLim = 2;
		while (!isTLE() && abs(ans_value) < INF) {
			int tmp_value = -INF;
			Point tmp = ans; // Instead of {0, 0}, never say die.

			unsigned vised = 0;
			for (auto i : choice) {
				Main.set(i, my);
				int value = -dfs(StepLim, en, -INF, -tmp_value);
				Main.reset(i);	

				if (value == TLE) break;

				vised++;

				if (value > tmp_value) {
					tmp_value = value;
					tmp = i;
				}
			}
			if (vised == choice.size()) {
				ans = tmp;
				ans_value = tmp_value;
				StepLim++;
			}
		}

		return ans;
	}
};

class UCT {
	private:
	static constexpr int ROUNDS_PER_GAME = 10;
	static constexpr double C = 1.4;
	struct Node {
		Board board; stone hand;
		vector <Point> choice;
		vector <Node*> son;
		Node* fa;
		unsigned size, now;
		int N;
		double Q;
		inline double value(const int faN) const {
			return (double)(N - Q) / N + C * sqLOG[faN] / sq[N];
		}
		inline Node* BestChild() const {
			if (!now) return NULL;
			Node *ret = son[0]; double rv = 0;
			for (unsigned i = 0; i < now; i++) {
				double val = son[i]->value(N);
				if (val > rv) ret = son[i], rv = val; 
			}
			return ret;
		}
	};

	Node t[int(1e5)], *ptr, *root;

	inline Node* newNode(const Board &x, const stone hand, const Point step = {-1, -1}, Node* fa = NULL) {
		ptr->board = x;
		ptr->hand = hand;
		ptr->fa = fa;
		if (fa != NULL) ptr->board.set(step, Switch(hand));
		ptr->board.getValidPoints(hand, ptr->choice);
		ptr->size = (unsigned)ptr->choice.size(), ptr->now = 0;
		ptr->son.resize(ptr->size);
		ptr->N = 0, ptr->Q = 0;
		return ptr++;
	}

	inline void update(Node *x, double Y) {
		while (x != NULL) {
			x->N += ROUNDS_PER_GAME;
			x->Q += Y;
			Y = ROUNDS_PER_GAME - Y;
			x = x->fa;
		}
	}

	inline Node* expand(Node *x) {
		x->son[x->now] = newNode(x->board, Switch(x->hand), x->choice[x->now], x);
		update(x->son[x->now], ROUNDS_PER_GAME * Rate(lasteval));
		return x->son[x->now++];
	}

	inline Node* policy(Node *x) {
		while (x->size) {
			if (x->now < x->size) return expand(x);
			else x = x->BestChild();
		}
		return x;
	}

	inline double simulation(const Node *x) {
		static Board tmp;
		static vector <Point> choice;
		int T = ROUNDS_PER_GAME; double Y = 0;
		while (T--) {
			stone hand = x->hand;
			tmp = x->board;
			while (true) {
				tmp.getValidPoints(hand, choice);
				if (!choice.size()) {
					Y += int(hand != x->hand);
					break;
				}
				tmp.set(choice[0], hand);
				hand = Switch(hand);
			}
		}
		return Y;
	}
	
	public:
	inline pair<int, int> Action() {
		isUCT = 1;
		ptr = t; 
		root = newNode(Main, my);

		// UCT Search
		while (!isTLE()) {
			Node *x = policy(root);
			double result = simulation(x);
			update(x, result);
		}
		Node *bc = root->BestChild();
		Point ans;
		for (unsigned i = 0; i < root->now; i++) {
			if (root->son[i] == bc) {
				ans = root->choice[i];
				break;
			}
		}
		return ans;
	}
};


Alpha_Beta alphabeta;
UCT uct;

inline void init() {
	cerr << "GLHF" << endl;
	if (ai_side == 0) my = BLACK, en = WHITE;
	else my = WHITE, en = BLACK;
	// precalculate
	/**** XXX : Would RE if enemy is too weak ****/	
	for (int i = 0; i < 100; i++) EXP[i] = exp(-i) / 2;
	for (int i = 1; i < 300000; i++) sqLOG[i] = (double)sqrtl(log(i));
	for (int i = 1; i < 300000; i++) sq[i] = (double)sqrtl(i);
}

inline void GetUpdate(pair<int, int> location) {
	Step++;
	vector <Point> choice;
	Main.getValidPoints(en, choice);	
	Main.set(location, en);
}

inline pair<int, int> Action() {
	Step++;
	StartTime = clock();

	if (Step == 1) { // First move of BLACK
		Main.set(make_pair(0, 0), my);
		return make_pair(0, 0);
	}

	vector <Point> choice;
	Main.getValidPoints(my, choice);

	Point ans;
	if (choice.size()) {
		ans = (9 < choice.size() && choice.size() < 27) ? uct.Action() : alphabeta.Action(); // divide-and-conqure on algorithms
	}
	else {  // Never say die!
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) if (!Main.at(i, j)) {
				choice.push_back({i, j});
			}
		}
		if (choice.size()) ans = choice[Rand() % (choice.size())];
	}

	Main.set(ans, my);
	return ans;
}

