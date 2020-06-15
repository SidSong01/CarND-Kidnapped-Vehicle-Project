/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100; // Set the number of particles

  // Create gaussian distribution for x, y, theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i<num_particles; i++) {
    Particle p;
    p.id = i;
    p.weight = 1.0;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta =dist_theta(gen);

    particles.push_back(p);
    weights.push_back(p.weight);
  }
  // set initialization as done
  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  normal_distribution<double> dist_x1(0, std_pos[0]);
  normal_distribution<double> dist_y1(0, std_pos[1]);
  normal_distribution<double> dist_theta1(0, std_pos[2]);
  
  for (int i = 0; i<num_particles;i++){
    // yaw rate is not 0
    if(fabs(yaw_rate)<0.001) {
      particles[i].x += velocity*cos(particles[i].theta)*delta_t;
      particles[i].y += velocity*sin(particles[i].theta)*delta_t;
    }
    // yaw rate is close to 0 
    else {
      particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate*delta_t;
    }
    // noise
    particles[i].x += dist_x1(gen);
    particles[i].y += dist_y1(gen);
    particles[i].theta += dist_theta1(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(unsigned int i =0; i < observations.size();i++) {
    LandmarkObs obser = observations[i];
    double min_d = 10000.0;
    int id_close = -1;
    int predict_index;

    for(unsigned int j = 0; j < predicted.size(); j++) {
      LandmarkObs predict = predicted[j];
      double d = (obser.x - predict.x) * (obser.x - predict.x) + (obser.y - predict.y) * (obser.y - predict.y);
      if(d < min_d) {
        min_d = d;
        id_close = predict.id;
        predict_index = j;
      }
    }
    observations[i].id = id_close;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double n_sum = 0.0;
  for(int i = 0; i < num_particles; i++) {
    // particles infomation
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    vector<LandmarkObs> predict;
    particles[i].weight = 1.0;
    
    // coordinate system transfer from Vehicle to Map
    for(unsigned int j=0; j<map_landmarks.landmark_list.size();j++) {
      Map::single_landmark_s lm = map_landmarks.landmark_list[j];
      double land_x = lm.x_f;
      double land_y = lm.y_f;
      double land_id = lm.id_i;
      double distence = (p_x - land_x)*(p_x - land_x)+(p_y - land_y)*(p_y - land_y);
      if(distence <= sensor_range*sensor_range) {
        LandmarkObs pre;
        pre.x = land_x;
        pre.y = land_y;
        pre.id = land_id;
        predict.push_back(pre);
      }
    }
    vector<LandmarkObs> observations_trans;
    // landmarks that have the distance between particle less than the sensor_range
    for(unsigned int k=0; k<observations.size(); k++) {
      double ob_x = observations[k].x;
      double ob_y = observations[k].y;
      int ob_id = observations[k].id;
      double tr_y = p_y +  sin(p_theta)*ob_x + cos(p_theta)*ob_y;
      double tr_x = p_x + cos(p_theta)*ob_x - sin(p_theta)*ob_y;
      observations_trans.push_back(LandmarkObs{ob_id, tr_x, tr_y});
    }
    
    // associate landmarks in range to landmark obeservations
    dataAssociation(predict, observations_trans);


    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double nomer = 1.0/(2*M_PI*std_x*std_y);
    for(unsigned int l=0; l< observations_trans.size(); l++) {
      double ob_tr_x = observations_trans[l].x;
      double ob_tr_y = observations_trans[l].y;
      int ob_tr_id = observations_trans[l].id;
      
      // update weights
      for(unsigned int m=0; m<predict.size(); m++) {
        double pred_x = predict[m].x;
        double pred_y = predict[m].y;
        double pred_id = predict[m].id;
        if(ob_tr_id == pred_id) {
          double difer_x = (ob_tr_x - pred_x)*(ob_tr_x - pred_x);
          double difer_y = (ob_tr_y - pred_y)*(ob_tr_y - pred_y);
          double new_weight = nomer*exp(-1.0*(difer_x/(2*std_x*std_x) + difer_y/(2*std_y*std_y)));
          particles[i].weight *= new_weight;
        }
      }
    }
  n_sum += particles[i].weight;
  }
  // update particles
  for(int i = 0; i < num_particles; i++) {
    particles[i].weight /=n_sum;
    weights[i] =  particles[i].weight;
  }
}
 
void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  uniform_int_distribution<int> un_int_dist(0, num_particles-1);
  int index = un_int_dist(gen);
  double beta = 0.0;
  vector<Particle> n_particles;
  
  double wei_max = *max_element(weights.begin(), weights.end());
  for(int j=0; j<num_particles;j++) {
    uniform_real_distribution<double> un_real_dist(0.0, wei_max);
    beta += 2.0* un_real_dist(gen);
    while(beta > weights[index]) {
      beta -= weights[index];
      index = (index +1) %num_particles;
    }
    n_particles.push_back(particles[index]);
  }
  particles = n_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}