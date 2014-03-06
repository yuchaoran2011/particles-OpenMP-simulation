#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"
#include "omp.h"


using namespace std;

//
//  benchmarking program
//
struct bin_t
{
    vector<particle_t*> particles;
    int num_particles; 

    bin_t();
};



void init_bins( bin_t* bins ) 
{
  for( int i = 0; i < NUM_BINS; i++) 
  {
      bins[i].num_particles = 0;
      bins[i].particles.clear();
  }
}


bin_t::bin_t() 
  : num_particles(0)
{}



void bin_particle( particle_t &particle, bin_t* bins ) 
{
  int col = floor(particle.x/BIN_SIZE);
  int row = floor(particle.y/BIN_SIZE);

  int index = row*NUM_BINS_PER_SIDE + col;
  bins[index].particles.push_back( &particle );
  bins[index].num_particles++;
}



void apply_force2( particle_t &particle, particle_t &neighbor )
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}





// Computing forces between all points in the current bin and all points in neighboring bins
void apply_force_for_bin(bin_t* boxes, int r, int c) 
{
  
  int i = r * NUM_BINS_PER_SIDE + c; // Bin index
  int box_size = NUM_BINS_PER_SIDE;

  for(int p = 0; p < boxes[i].num_particles; p++) 
  {

        (*boxes[i].particles[p]).ax = 0;
        (*boxes[i].particles[p]).ay = 0;

        for(int p2 = 0; p2 < boxes[i].num_particles; p2++) 
        {
            apply_force2( *boxes[i].particles[p], *boxes[i].particles[p2] );
        }

        if( c-1 >= 0 ) 
        {
            for(int p2 = 0; p2 < boxes[i-1].num_particles; p2++) 
            {
                apply_force2( *boxes[i].particles[p], *boxes[i-1].particles[p2] );
            }
        }

        if( c+1 < NUM_BINS_PER_SIDE )
        {
            for(int p2 = 0; p2 < boxes[i+1].num_particles; p2++) 
            {
                apply_force2( *boxes[i].particles[p], *boxes[i+1].particles[p2] );
            }
        }

        if (r-1 >= 0)
        {
            for(int p2 = 0; p2 < boxes[i-box_size].num_particles; p2++) 
            {
                apply_force2( *boxes[i].particles[p], *boxes[i-box_size].particles[p2] );
            }
        }

        if (r+1 < NUM_BINS_PER_SIDE)
        {
            for(int p2 = 0; p2 < boxes[i+box_size].num_particles; p2++) 
            {
                apply_force2( *boxes[i].particles[p], *boxes[i+box_size].particles[p2] );
            }
        }

        if ((c - 1 >= 0) && (r-1 >= 0))
        {
            for(int p2 = 0; p2 < boxes[i-box_size-1].num_particles; p2++) 
            {
                apply_force2( *boxes[i].particles[p], *boxes[i-box_size-1].particles[p2] );
            }
        }

        if ((c + 1 < NUM_BINS_PER_SIDE) && (r-1 >= 0))
        {
            for(int p2 = 0; p2 < boxes[i-box_size+1].num_particles; p2++) 
            {
                apply_force2( *boxes[i].particles[p], *boxes[i-box_size+1].particles[p2] );
            }
        }

        if ((c - 1 >= 0) && (r+1 < NUM_BINS_PER_SIDE))
        {
            for(int p2 = 0; p2 < boxes[i+box_size-1].num_particles; p2++) 
            {
                apply_force2( *boxes[i].particles[p], *boxes[i+box_size-1].particles[p2] );
            }
        }

        if ((c + 1 < NUM_BINS_PER_SIDE) && (r+1 < NUM_BINS_PER_SIDE))
        {
            for(int p2 = 0; p2 < boxes[i+box_size+1].num_particles; p2++) 
            {
                apply_force2( *boxes[i].particles[p], *boxes[i+box_size+1].particles[p2] );
            }
        }
    }
}






int main( int argc, char **argv )
{   
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );

    assert( bin_size > 2*cutoff );

    printf("%d\n", n);

    init_particles( n, particles );

    bin_t* bins = new bin_t[NUM_BINS];
    init_bins( bins ); 


    for( int i = 0; i < n; i++ ) { 
        bin_particle(particles[i], bins);
    }
    
    double simulation_time = read_timer( );

    #pragma omp parallel
    for( int step = 0; step < NSTEPS; step++ )
    {
        #pragma omp for
        for(int i = 0; i < NUM_BINS; i++) {
            int r = i/NUM_BINS_PER_SIDE;
            int c = i % NUM_BINS_PER_SIDE;
            apply_force_for_bin(bins, r, c); 
        }

        #pragma omp for
        for( int i = 0; i < n; i++ ) 
        {
            move( particles[i] );
        }

        #pragma omp single
        {
            init_bins( bins ); 
            for( int i = 0; i < n; i++ ) { 
                bin_particle(particles[i], bins);
            }

            if( fsave && (step%SAVEFREQ) == 0 )
            {
                save( fsave, n, particles );
            }
        }
    }

    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    init_bins( bins );   
    free( particles );
    delete[] ( bins );
    if( fsave ) {
        fclose( fsave );
    }
    
    return 0;
}
