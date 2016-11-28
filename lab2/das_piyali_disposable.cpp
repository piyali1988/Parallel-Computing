#include<iostream>
#include<cstdio>
#include<stdlib.h>
#include<string>
#include<math.h>
#include<stack>
#include<fstream>
#include<sstream>
#include<vector>
#include<functional>
#include <ctime>
#include <chrono>
#include <numeric>
#include<pthread.h>

using namespace std;

#define MAX_ITERATIONS 100000000

double AFFECT_RATE;
double EPSILON;
int NUM_THREADS;

struct grid_block{
	int box_id;
	int up_left_x;
	int up_left_y;
	int height;
	int width;
	int num_top_n;
	int num_bottom_n;
	int num_left_n;
	int num_right_n;
	vector<int> top_n;
	vector<int> bottom_n;
	vector<int> left_n;
	vector<int> right_n;
	double temperature;
	int perimeter;
};

vector<grid_block> grid_blocks;
vector<double> temporary;
pthread_mutex_t mut;
//int g_num_of_threads = NUM_THREADS;
int g_num_of_grids;

void print_block(grid_block box);

void print_blocks(vector<grid_block>& grid_blocks);

void parse_input(vector<grid_block>& grid_blocks);

double sum_of_temp_on_perimeter(grid_block box,vector<grid_block>& grid_blocks);

double contact_dist(int id1, int id2, vector<grid_block>& grid_blocks);

double temperature_difference(int id1,vector<grid_block>& grid_blocks);

void effective_perimeter(vector<grid_block>& grid_blocks);

void DSV_iteration(vector<grid_block>& grid_blocks);

void *compute_and_store (void *);

int main(int argc, char **argv){
	clock_t begin = clock();
	AFFECT_RATE = atof(argv[1]);
	EPSILON = atof(argv[2]);
	NUM_THREADS = atoi(argv[3]);
	cout<<"Affect_rate : "<<argv[1]<<endl;
	cout<<"Epsilon : "<<argv[2]<<endl;
	cout<<"Threads : "<<argv[3]<<endl;

	int num_of_grids,num_grid_rows,num_grid_columns;	
	string line;
	int id;
//	vector<grid_block> grid_blocks;
	//parse input from the file.
	parse_input(grid_blocks);
	//print the parsed data.
	//print_blocks(grid_blocks);
	effective_perimeter(grid_blocks);
	DSV_iteration(grid_blocks);
	clock_t end = clock();
  	double elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
  	cout<<"Complete running time : "<<elapsed_secs<<endl;
	return 0;
}

void parse_input(vector<grid_block>& grid_blocks ){
	int num_of_grids,num_grid_rows,num_grid_columns;
	int id;	
	string line;

	    	cin>>g_num_of_grids;
	    	cin>>num_grid_rows;
	    	cin>>num_grid_columns;
		num_of_grids = g_num_of_grids;
				
		//Lets get the grid parameters here.
		for (int i=0;i<num_of_grids;i++){
			//get the grid box id.
			//add a new grid block to the array of blocks.
		  	grid_blocks.push_back(grid_block());
		  	cin >> grid_blocks[i].box_id;

		  	//get coordinates of the box and its size.
		  	cin >> grid_blocks[i].up_left_x;
		  	cin >> grid_blocks[i].up_left_y;
		  	cin >> grid_blocks[i].height;
		  	cin >> grid_blocks[i].width;
		  	
		  	//get the details about neighbors.
		  	//get the details of the top neighbors.
		  	cin >> grid_blocks[i].num_top_n;
		  	for (int j=0;j<grid_blocks[i].num_top_n;j++){
		  		cin >> id;
		  		grid_blocks[i].top_n.push_back(id);
			}
			//get details of the bottom neighnbors.
		  	cin >> grid_blocks[i].num_bottom_n;
		  	for (int j=0;j<grid_blocks[i].num_bottom_n;j++){
		  		cin >> id;
		  		grid_blocks[i].bottom_n.push_back(id);
			}
			//get details of the right neighnbors.
		  	cin >> grid_blocks[i].num_left_n;
		  	for (int j=0;j<grid_blocks[i].num_left_n;j++){
		  		cin >> id;
		  		grid_blocks[i].left_n.push_back(id);
			}
			//get details of the left neighnbors.
		  	cin>> grid_blocks[i].num_right_n;
		  	for (int j=0;j<grid_blocks[i].num_right_n;j++){
		  		cin >> id;
		  		grid_blocks[i].right_n.push_back(id);
			}
			//get the temperature of the box.
			cin >> grid_blocks[i].temperature;
			//print the grid block.
			//print_block(grid_blocks[i]);
		}
}

void print_blocks(vector<grid_block>& grid_blocks){
	for (int i=0;i<grid_blocks.size();i++){
		print_block(grid_blocks[i]);
	}
}

void print_block(grid_block box){
	cout<<"--------------"<<endl;
	cout<<"box_id : "<<box.box_id<<endl;
	cout<<"up_left_x : "<<box.up_left_x<<" up_left_y : "<<box.up_left_y<<" height : "<<box.height<<" width : "<<box.width<<endl;
	cout<<"num_top_n : "<<box.num_top_n<<" ::: ";
	for (int i=0;i<box.num_top_n;i++){
		cout<<box.top_n[i]<<" , ";
	}
	cout<<endl;
	cout<<"num_bottom_n : "<<box.num_bottom_n<<" ::: ";
	for (int i=0;i<box.num_bottom_n;i++){
		cout<<box.bottom_n[i]<<" , ";
	}
	cout<<endl;
	cout<<"num_left_n : "<<box.num_left_n<<" ::: ";
	for (int i=0;i<box.num_left_n;i++){
		cout<<box.left_n[i]<<" , ";
	}
	cout<<endl;
	cout<<"num_right_n : "<<box.num_right_n<<" ::: ";
	for (int i=0;i<box.num_right_n;i++){
		cout<<box.right_n[i]<<" , ";
	}
	cout<<endl;
	cout<<"temperature : "<<box.temperature<<endl;
	cout<<"perimeter : "<<box.perimeter<<endl;
	cout<<"--------------"<<endl;
}

double sum_of_temp_on_perimeter(grid_block box,vector<grid_block>& grid_blocks){
	double temperature = 0.0;
	int id;
	//Add temperatures on the top neighbors.
	for (int i=0;i<box.num_top_n;i++){
		id = box.top_n[i];
		temperature += contact_dist(box.box_id,id,grid_blocks)*grid_blocks[id].temperature;
	}
	//Add temperatures on the bottom neighbors.
	for (int i=0;i<box.num_bottom_n;i++){
		id = box.bottom_n[i];
		temperature += contact_dist(box.box_id,id,grid_blocks)*grid_blocks[id].temperature;
	}
	//Add temperatures on the left neighbors.
	for (int i=0;i<box.num_left_n;i++){
		id = box.left_n[i];
		temperature += contact_dist(box.box_id,id,grid_blocks)*grid_blocks[id].temperature;
	}
	//Add temperatures on the right neighbors.
	for (int i=0;i<box.num_right_n;i++){
		id = box.right_n[i];
		temperature += contact_dist(box.box_id,id,grid_blocks)*grid_blocks[id].temperature;
	}
	return temperature;
}

double contact_dist(int id1, int id2, vector<grid_block>& grid_blocks){
	double weight = 0;
	if((grid_blocks[id1].up_left_x <= grid_blocks[id2].up_left_x) && (grid_blocks[id2].up_left_x < (grid_blocks[id1].up_left_x + grid_blocks[id1].width))){
		if((grid_blocks[id2].up_left_x + grid_blocks[id2].width) < (grid_blocks[id1].up_left_x + grid_blocks[id1].width)){
			weight =  grid_blocks[id2].width;
		}else{
			weight =  grid_blocks[id1].up_left_x + grid_blocks[id1].width - grid_blocks[id2].up_left_x;
		}
	}else if((grid_blocks[id2].up_left_x <= grid_blocks[id1].up_left_x ) && (grid_blocks[id1].up_left_x < (grid_blocks[id2].up_left_x + grid_blocks[id2].width))){
		if((grid_blocks[id1].up_left_x + grid_blocks[id1].width) < (grid_blocks[id2].up_left_x + grid_blocks[id2].width)){
			weight = grid_blocks[id1].width;
		}else{
			weight = grid_blocks[id2].up_left_x + grid_blocks[id2].width - grid_blocks[id1].up_left_x;
		}
	}else if((grid_blocks[id1].up_left_y <= grid_blocks[id2].up_left_y ) && ( grid_blocks[id2].up_left_y < (grid_blocks[id1].up_left_y + grid_blocks[id1].height))){
		if((grid_blocks[id2].up_left_y + grid_blocks[id2].height) < (grid_blocks[id1].up_left_y + grid_blocks[id1].height)){
			weight = grid_blocks[id2].height;
		}else{
			weight = grid_blocks[id1].up_left_y + grid_blocks[id1].height - grid_blocks[id2].up_left_y;
		}
	}else if((grid_blocks[id2].up_left_y <= grid_blocks[id1].up_left_y ) && (grid_blocks[id1].up_left_y < (grid_blocks[id2].up_left_y + grid_blocks[id2].height))){
		if((grid_blocks[id1].up_left_y + grid_blocks[id1].height) < (grid_blocks[id2].up_left_y + grid_blocks[id2].height)){
			weight = grid_blocks[id1].height;
		}else{
			weight = grid_blocks[id2].up_left_y + grid_blocks[id2].height - grid_blocks[id1].up_left_y;
		}
	}
	//cout<<"weight-----"<<weight<<endl;
	return weight;
}

double temperature_difference(int id,vector<grid_block>& grid_blocks){
	double diff;
	diff = grid_blocks[id].temperature - (sum_of_temp_on_perimeter(grid_blocks[id],grid_blocks)/grid_blocks[id].perimeter);
	return diff;
}

void effective_perimeter(vector<grid_block>& grid_blocks){
	int id;
	for(int i=0;i<grid_blocks.size();i++){
		int perimeter = 0;
		for (int j=0;j<grid_blocks[i].num_top_n;j++){
			id = grid_blocks[i].top_n[j];
			perimeter += contact_dist(grid_blocks[i].box_id,id,grid_blocks);
		}
		//Add temperatures on the bottom neighbors.
		for (int j=0;j<grid_blocks[i].num_bottom_n;j++){
			id = grid_blocks[i].bottom_n[j];
			perimeter += contact_dist(grid_blocks[i].box_id,id,grid_blocks);
		}
		//Add temperatures on the left neighbors.
		for (int j=0;j<grid_blocks[i].num_left_n;j++){
			id = grid_blocks[i].left_n[j];
			perimeter += contact_dist(grid_blocks[i].box_id,id,grid_blocks);
		}
		//Add temperatures on the right neighbors.
		for (int j=0;j<grid_blocks[i].num_right_n;j++){
			id = grid_blocks[i].right_n[j];
			perimeter += contact_dist(grid_blocks[i].box_id,id,grid_blocks);
		}
		grid_blocks[i].perimeter = perimeter;
	}
}

void DSV_iteration(vector<grid_block>& grid_blocks){
	//vector<double> temporary;
	bool stop = false;
	for(int i=0;i<grid_blocks.size();i++){
		temporary.push_back(0);
	}
	int j=0;
	double min,max;
	auto t_chrono = chrono::system_clock::now();
	clock_t begin = clock();
	while(j<MAX_ITERATIONS && !stop)
	{
		pthread_t threads[NUM_THREADS];	
		for(int i=0; i < NUM_THREADS; i++ ){
      			pthread_create(&threads[i], NULL , compute_and_store,(void *)i);
  		}
		for(int i=0; i < NUM_THREADS; i++ ){
   			pthread_join(threads[i], NULL );
  		}
		
		/*for(int i=0;i<grid_blocks.size();i++){
			double diff = temperature_difference(i,grid_blocks);
			temporary[i] = grid_blocks[i].temperature - diff*AFFECT_RATE;
		}*/
		min = temporary[0];
		max = temporary[0];
		for(int i=0;i<grid_blocks.size();i++){
			grid_blocks[i].temperature = temporary[i];
			if(temporary[i] < min){
				min = temporary[i];
			}
			if(temporary[i] > max){
				max = temporary[i];
			}
		}
		j++;
		if((max-min) < max*EPSILON){
			cout<<"min temperature : "<<min<<endl<<"max temperature : "<<max<<endl;
			stop = true;
		} 
	}
	auto t_chronod = chrono::system_clock::now() - t_chrono;
	cout<<"final iteration : "<<j<<endl;
	clock_t end = clock();
  	double actual_time = double(end - begin)/CLOCKS_PER_SEC;
  	cout<<"Loop running time : "<< actual_time <<endl;
	cout  << "Elapsed convergence loop time (chrono): "<<chrono::duration<double,std::milli>(t_chronod).count() << endl;
}

void *compute_and_store (void* i){
	int k = *((int*) (&i));
	for (int j=k;j<g_num_of_grids;j+=NUM_THREADS){
		double diff = temperature_difference(j,grid_blocks);
		temporary[j] = grid_blocks[j].temperature - diff*AFFECT_RATE;
		//cout<<"grid number called : "<<j<<endl;
	}
	pthread_exit(&k);
}

