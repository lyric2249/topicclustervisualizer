
#include <string>
using namespace std;

void Model::load_docs(string dfile) 
{
  //cout << "load docs: " << dfile << endl;
  ifstream rf( dfile.c_str() );
  if (!rf) {
	cout << "file not find:" << dfile << endl;
	exit(-1);
  }

  string line;
  while(getline(rf, line)) {
	Doc doc(line);
	doc.gen_biterms(bs);
	// statistic the exmperial word distribution
	for (int i = 0; i < doc.size(); ++i) {
	  int w = doc.get_w(i);
	  pw_b[w] += 1;
	}
  }
  
  pw_b.normalize();
}


