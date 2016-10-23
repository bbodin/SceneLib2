/*
 * monoslam_library.cpp
 *
 *  Created on: 21 Oct 2016
 *      Author: toky
 */

#include<SLAMBenchAPI.h>


#include <stdint.h>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <csignal>
#include "monoslam.h"
#include "kalman.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>

#include <MonoSLAMApplication.h>
#include <timings.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>




void init(SceneLib2::MonoSLAM* monoslam, MonoSLAMApplication* config)
{


    /***
     *
     * Variables that I don't manage well yet
     *   * some camera parameters
     *   * and the initial position parameters
     *   + some constant that are probably supposed to stay ... (later in the code)
     *
     */


    double state_vw_x = 0.0;
    double state_vw_y = 0.0;
    double state_vw_z = -0.1;

    double state_ww_x = 0.0;
    double state_ww_y = 0.0;
    double state_ww_z = 0.01;

    // Setting of the initial position and the quaternion

    Eigen::Vector3f init_pose =   {config->get_initial_pos_factor().x,config->get_initial_pos_factor().y,config->get_initial_pos_factor().z};

    double state_rw_x = init_pose[0];
    double state_rw_y = init_pose[0];
    double state_rw_z = init_pose[1];

    double state_qwr_x = config->get_init_pose_rotation().x;
    double state_qwr_y = config->get_init_pose_rotation().y;
    double state_qwr_z = config->get_init_pose_rotation().z;
    double state_qwr_w = config->get_init_pose_rotation().w;



    // Keep this order!
    //  1. Camera
    //  2. MotionModel
    //  3. FullFeatureModel and PartFeatureModel

    const sb_uint2 inputSize = config->get_input_size();
    std::cerr << "input Size is = " << inputSize.x << "," << inputSize.y << std::endl;

    sb_float4 camera =   config->get_camera();

    camera.x = camera.x *  inputSize.x;
    camera.y = camera.y *  inputSize.y;
    camera.z = camera.z *  inputSize.x;
    camera.w = camera.w *  inputSize.y;

    std::cerr << "camera is = " << camera.x  << "," << camera.y  << "," << camera.z  << "," << camera.w
            << std::endl;

    monoslam->camera_ = new SceneLib2::Camera();
    monoslam->camera_->SetCameraParameters(inputSize.x, inputSize.y,
                                           camera.x, camera.z, camera.y, camera.w,
                                           config->cam_kd1, config->cam_sd);

    monoslam->motion_model_ = new SceneLib2::MotionModel();

    monoslam->full_feature_model_ = new SceneLib2::FullFeatureModel(2, 3, 3, monoslam->camera_, monoslam->motion_model_);
    monoslam->part_feature_model_ = new SceneLib2::PartFeatureModel(2, 6, 6, monoslam->camera_, monoslam->motion_model_, 3);

    // Initialise constant-like variables
    monoslam->kDeltaT_ = config->delta_t;
    monoslam->kNumberOfFeaturesToSelect_ = config->number_of_features_to_select;
    monoslam->kNumberOfFeaturesToKeepVisible_ = config->number_of_features_to_keep_visible;
    monoslam->kMaxFeaturesToInitAtOnce_ = config->max_features_to_init_at_once;
    monoslam->kMinLambda_ = config->min_lambda;
    monoslam->kMaxLambda_ = config->max_lambda;
    monoslam->kNumberOfParticles_ = config->number_of_particles;
    monoslam->kStandardDeviationDepthRatio_ = config->standard_deviation_depth_ratio;
    monoslam->kMinNumberOfParticles_ = config->min_number_of_particles;
    monoslam->kPruneProbabilityThreshold_ = config->prune_probability_threshold;
    monoslam->kErasePartiallyInitFeatureAfterThisManyAttempts_ = config->erase_partially_init_feature_after_this_many_attempts;

    monoslam->minimum_attempted_measurements_of_feature_ = config->minimum_attempted_measurements_of_feature;
    monoslam->successful_match_fraction_ = config->successful_match_fraction;

    /**
     *
     * The constants
     *
     */

    monoslam->number_of_visible_features_ = 0;
    monoslam->next_free_label_ = 0;
    monoslam->marked_feature_label_ = -1;
    monoslam->total_state_size_ = monoslam->motion_model_->kStateSize_;

    monoslam->xv_.resize(monoslam->motion_model_->kStateSize_);
    monoslam->xv_ << state_rw_x, state_rw_y, state_rw_z,
                     state_qwr_w, state_qwr_x, state_qwr_y, state_qwr_z,
                     state_vw_x, state_vw_y, state_vw_z,
                     state_ww_x, state_ww_y, state_ww_z;

    monoslam->Pxx_.resize(monoslam->motion_model_->kStateSize_, monoslam->motion_model_->kStateSize_);
    monoslam->Pxx_.setZero(monoslam->motion_model_->kStateSize_, monoslam->motion_model_->kStateSize_);

    // Good for the last step, ready for capturing, drawing and processing
    monoslam->kalman_ = new SceneLib2::Kalman();

    monoslam->init_feature_search_region_defined_flag_ = false;
    monoslam->location_selected_flag_ = false;

    srand48(0); // Always the same seed (pick a number), so deterministic
}

SceneLib2::MonoSLAM *g_monoslam;
MonoSLAMApplication* config;
cv::Mat * inputGrey;


void sb_new_slam_configuration(SLAMBenchConfiguration ** slam_settings) {
    *slam_settings = new MonoSLAMApplication();
}

bool sb_init_slam_system(SLAMBenchConfiguration * slam_settings) {

    config = dynamic_cast<MonoSLAMApplication*>(slam_settings);

    Eigen::Vector3f init_pose =   {config->get_initial_pos_factor().x,config->get_initial_pos_factor().y,config->get_initial_pos_factor().z};
    const Eigen::Vector2i inputSize = {config->get_input_size().x,config->get_input_size().y};
    std::cerr << "input Size is = " << inputSize[0] << "," << inputSize[1] << std::endl;

    //  =========  BASIC PARAMETERS  (input size / computation size )  =========





    //  =========  BASIC BUFFERS  (input / output )  =========


    // Generate the initial position matrix
    Eigen::Matrix<float, 4,4> mInit(4,4);
    mInit << 1, 0, 0, init_pose[0],
            0, 1, 0, init_pose[1],
            0, 0, 1, init_pose[2],
            0, 0, 0, 1;

    Eigen::Vector4f vals = {
            config->get_init_pose_rotation().x,
            config->get_init_pose_rotation().y,
            config->get_init_pose_rotation().z,
            config->get_init_pose_rotation().w
    };

    Eigen::Quaternionf  initial_pose_quant = Eigen::Quaternionf(vals[3], vals[0], vals[1], vals[2]);
    mInit.block<3,3>(0,0) =  initial_pose_quant.toRotationMatrix();

    Eigen::Matrix<float, 4,4> poseMatrix;
    poseMatrix(0,0) = mInit.coeff(0, 0);
    poseMatrix(0,1) = mInit.coeff(0, 1);
    poseMatrix(0,2) = mInit.coeff(0, 2);
    poseMatrix(0,3) = mInit.coeff(0, 3);

    poseMatrix(1,0) = mInit.coeff(1, 0);
    poseMatrix(1,1) = mInit.coeff(1, 1);
    poseMatrix(1,2) = mInit.coeff(1, 2);
    poseMatrix(1,3) = mInit.coeff(1, 3);

    poseMatrix(2,0) = mInit.coeff(2, 0);
    poseMatrix(2,1) = mInit.coeff(2, 1);
    poseMatrix(2,2) = mInit.coeff(2, 2);
    poseMatrix(2,3) = mInit.coeff(2, 3);

    poseMatrix(3,0) = mInit.coeff(3, 0);
    poseMatrix(3,1) = mInit.coeff(3, 1);
    poseMatrix(3,2) = mInit.coeff(3, 2);
    poseMatrix(3,3) = mInit.coeff(3, 3);

    std::cerr << "Init pos is = " << poseMatrix(0,0)  << "," << poseMatrix(0,1)  << "," << poseMatrix(0,2) << "," << poseMatrix(0,3) << std::endl;
    std::cerr << "Init pos is = " << poseMatrix(1,0)  << "," << poseMatrix(1,1)  << "," << poseMatrix(1,2) << "," << poseMatrix(1,3) << std::endl;
    std::cerr << "Init pos is = " << poseMatrix(2,0)  << "," << poseMatrix(2,1)  << "," << poseMatrix(2,2) << "," << poseMatrix(2,3) << std::endl;
    std::cerr << "Init pos is = " << poseMatrix(3,0)  << "," << poseMatrix(3,1)  << "," << poseMatrix(3,2) << "," << poseMatrix(3,3) << std::endl;

    g_monoslam = new SceneLib2::MonoSLAM ();

    init(g_monoslam, config);

    inputGrey = new cv::Mat ( inputSize[1] ,  inputSize[0], CV_8UC1);


    return true;
}

bool sb_update_frame (void * data, frame_format type) {
    switch (type) {
        case GREY_FRAME :
            memcpy ( (unsigned char*)inputGrey->data , data , sizeof(unsigned char) * config->get_input_size().x * config->get_input_size().y);
            return true;
        default :
            return false;
       };


}

bool sb_process_once () {
    double timings[3];
    timings[0] = tock();

     g_monoslam->GoOneStep(*inputGrey,
             false,
             true);


     timings[1] = tock();
     config->sample("computation",  timings[1] - timings[0],PerfStats::Type::TIME);
     config->sample("total-features",     g_monoslam->feature_list_.size(),PerfStats::Type::COUNT);
     config->sample("visible-features",     g_monoslam->number_of_visible_features_,PerfStats::Type::COUNT);
     config->sample("total-state-size",     g_monoslam->total_state_size_,PerfStats::Type::COUNT);
     config->sample("marked-feature-label",     g_monoslam->marked_feature_label_,PerfStats::Type::COUNT);
     config->sample("next-free-label",     g_monoslam->next_free_label_,PerfStats::Type::COUNT);

     config->sample("minimum-attempted",     g_monoslam->minimum_attempted_measurements_of_feature_,PerfStats::Type::COUNT);
     config->sample("successful-match",     g_monoslam->successful_match_fraction_,PerfStats::Type::COUNT);


     return true;

}

sb_float3  sb_get_position     () {
    Eigen::Matrix<float, 4,4> pose = g_monoslam->getPose();



     float x = pose(0,3);
     float y = pose(1,3);
     float z = pose(2,3);

     return {x,y,z};
}
bool       sb_get_tracked  () {
    return g_monoslam->number_of_visible_features_ > 0;
}

bool sb_save_map (const char * , map_format){
    return false;
}

bool sb_clean_slam_system(){
    return true;
}


bool sb_initialize_ui(SLAMBenchUI * ui) {
	ui->init(0, GREY8,  config->get_input_size().x, config->get_input_size().y);
    return true;
}

bool sb_update_ui(SLAMBenchUI * ui) {
	ui->update(0, inputGrey->data);
    return true;
}
