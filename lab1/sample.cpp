#include<iostream>
#include<cstdio>
#include<stdlib.h>
#include<string>
#include<math.h>
#include<stack>
#include<fstream>
#include<sstream>
#include<vector>
using namespace std;

int main(int argc, char **argv){

	int num_of_grids,num_grid_rows,num_grid_columns,x,y;
	cin >> num_of_grids;
	cin >> num_grid_rows;
	cin >> num_grid_columns;
cout<<"grids"<<num_of_grids<<"\trows"<<num_grid_rows<<"\tCols"<<num_grid_columns;
	for (int i=0;i<num_of_grids;i++){

		cin>>x;
		cin>>y;
cout<<"\nX"<<x<<"\tY"<<y<<endl;

	}

return 0;



}
