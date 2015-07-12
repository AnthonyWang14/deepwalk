#include <fstream>
using namespace std;

int main(int argc, char const *argv[]) {
	/* code */
	ifstream in("karate.adjlist", ios::in);
	ofstream out("new_karate.adjlist", ios::out);
	int i = 0;
	while (in >> i) {
		out << i-1 << endl;
	}
	return 0;
}