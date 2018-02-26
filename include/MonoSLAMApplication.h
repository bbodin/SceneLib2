/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef MONOSLAM_APPLICATION_H_
#define MONOSLAM_APPLICATION_H_

#include <vector>
#include <sstream>

#include <SLAMBenchConfiguration.h>



////////////////////////// RUNTIME PARAMETERS //////////////////////

    const double default_cam_kd1 = 9e-06;
    const int    default_cam_sd  = 1;

    const double default_delta_t = 0.033333333;
    const int    default_number_of_features_to_select = 10;
    const int    default_number_of_features_to_keep_visible = 12;
    const int    default_max_features_to_init_at_once = 1;
    const double default_min_lambda = 0.5;
    const double default_max_lambda = 5.0;
    const int    default_number_of_particles = 100;
    const double default_standard_deviation_depth_ratio = 0.3;
    const int    default_min_number_of_particles = 20;
    const double default_prune_probability_threshold = 0.05;
    const int    default_erase_partially_init_feature_after_this_many_attempts = 10;
    const int    default_minimum_attempted_measurements_of_feature = 20;
    const double default_successful_match_fraction = 0.5;

class MonoSLAMApplication : public SLAMBenchConfiguration {

	// Possible Parameters
public :
    double delta_t;
    int    number_of_features_to_select;
    int    number_of_features_to_keep_visible;
    int    max_features_to_init_at_once;
    double min_lambda;
    double max_lambda;
    int    number_of_particles;
    double standard_deviation_depth_ratio;
    int    min_number_of_particles;
    double prune_probability_threshold;
    int    erase_partially_init_feature_after_this_many_attempts;
    double cam_kd1;
    int    cam_sd;
    int    minimum_attempted_measurements_of_feature;
    double successful_match_fraction;
public :
    MonoSLAMApplication () : SLAMBenchConfiguration() {
    	// Free : c d e g h j l m n r s t u v w x y z

    }

    ~MonoSLAMApplication() {}

};

#endif /* MONOSLAM_APPLICATION_H_ */
