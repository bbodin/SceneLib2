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
#include "io/sensor/CameraSensorFinder.h"
#include "io/sensor/CameraSensor.h"
#include "io/SLAMFrame.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>

#include <MonoSLAMApplication.h>
#include <timings.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>


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

SceneLib2::MonoSLAM *g_monoslam;

cv::Mat * inputGrey;

const slambench::io::CameraSensor *grey_sensor = nullptr;

slambench::outputs::Output *pose_output = nullptr;
slambench::outputs::Output *frame_output = nullptr;
slambench::outputs::Output *frame_info_output = nullptr;
slambench::outputs::Output *features_output = nullptr;

slambench::outputs::Output *tracked_features_output = nullptr;
slambench::outputs::Output *init_features_output = nullptr;

const slambench::io::CameraSensor *GetSensor(SLAMBenchLibraryHelper* config) {
	slambench::io::CameraSensorFinder sensor_finder;
	const slambench::io::CameraSensor *sensor = sensor_finder.FindOne(config->get_sensors(), {{"camera_type", "grey"}});
    
	if(sensor == nullptr) {
		std::cerr << "Could not find greyscale sensor" << std::endl;
		exit(1);
	}
	
	if(sensor->PixelFormat != slambench::io::pixelformat::G_I_8) {
		std::cerr << "Grey sensor has wrong pixel format" << std::endl;
		exit(1);
	}

	if(sensor->FrameFormat != slambench::io::frameformat::Raster) {
		std::cerr << "Grey sensor has wrong frame format" << std::endl;
		exit(1);
	}
	
	return sensor;
}

void init(SceneLib2::MonoSLAM* monoslam, SLAMBenchLibraryHelper* config)
{
	grey_sensor = GetSensor(config);
	

    /***
     *
     * Variables that I don't manage well yet
     *   * some camera parameters
     *   * and the initial position parameters
     *   + some constant that are probably supposed to stay ... (later in the code)
     *
     */

	// TODO: The monoslam camera position is slightly modified? this needs to
	// be reflected in the final pose matrix (which at the moment comes 
	// straight from the sensor)
    
	double state_vw_x = 0.0;
    double state_vw_y = 0.0;
    double state_vw_z = -0.1;

    double state_ww_x = 0.0;
    double state_ww_y = 0.0;
    double state_ww_z = 0.01;

    // Setting of the initial position and the quaternion

    double state_rw_x = grey_sensor->Pose(0,3);
    double state_rw_y = grey_sensor->Pose(1,3);
    double state_rw_z = grey_sensor->Pose(2,3);

	// TODO: extract rotation quaternion from pose matrix
    double state_qwr_x = 0; //config->get_initial_rotation().x;
    double state_qwr_y = 0; //config->get_initial_rotation().y;
    double state_qwr_z = 0; //config->get_initial_rotation().z;
    double state_qwr_w = 1; //config->get_initial_rotation().w;
	
    // Keep this order!
    //  1. Camera
    //  2. MotionModel
    //  3. FullFeatureModel and PartFeatureModel

    const sb_uint2 inputSize = {(unsigned int)grey_sensor->Width,(unsigned int)grey_sensor->Height};
    std::cerr << "input Size is = " << inputSize.x << "," << inputSize.y << std::endl;

    sb_float4 camera =   {grey_sensor->Intrinsics[0], grey_sensor->Intrinsics[1], grey_sensor->Intrinsics[2], grey_sensor->Intrinsics[3]};

    camera.x = camera.x *  inputSize.x;
    camera.y = camera.y *  inputSize.y;
    camera.z = camera.z *  inputSize.x;
    camera.w = camera.w *  inputSize.y;

    std::cerr << "camera is = " << camera.x  << "," << camera.y  << "," << camera.z  << "," << camera.w
            << std::endl;

    monoslam->camera_ = new SceneLib2::Camera();
    monoslam->camera_->SetCameraParameters(inputSize.x, inputSize.y,
                                           camera.x, camera.z, camera.y, camera.w,
                                           cam_kd1, cam_sd);

    monoslam->motion_model_ = new SceneLib2::MotionModel();

    monoslam->full_feature_model_ = new SceneLib2::FullFeatureModel(2, 3, 3, monoslam->camera_, monoslam->motion_model_);
    monoslam->part_feature_model_ = new SceneLib2::PartFeatureModel(2, 6, 6, monoslam->camera_, monoslam->motion_model_, 3);

    // Initialise constant-like variables
    monoslam->kDeltaT_ = delta_t;
    monoslam->kNumberOfFeaturesToSelect_ = number_of_features_to_select;
    monoslam->kNumberOfFeaturesToKeepVisible_ = number_of_features_to_keep_visible;
    monoslam->kMaxFeaturesToInitAtOnce_ = max_features_to_init_at_once;
    monoslam->kMinLambda_ = min_lambda;
    monoslam->kMaxLambda_ = max_lambda;
    monoslam->kNumberOfParticles_ = number_of_particles;
    monoslam->kStandardDeviationDepthRatio_ = standard_deviation_depth_ratio;
    monoslam->kMinNumberOfParticles_ = min_number_of_particles;
    monoslam->kPruneProbabilityThreshold_ = prune_probability_threshold;
    monoslam->kErasePartiallyInitFeatureAfterThisManyAttempts_ = erase_partially_init_feature_after_this_many_attempts;

    monoslam->minimum_attempted_measurements_of_feature_ = minimum_attempted_measurements_of_feature;
    monoslam->successful_match_fraction_ = successful_match_fraction;

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

bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings) {

	        slam_settings->addParameter(TypedParameter<double> ("dt", "delta_t", "delta_t", &delta_t, &default_delta_t));
	        slam_settings->addParameter(TypedParameter<int   > ("fs", "features_select", "number_of_features_to_select", &number_of_features_to_select, &default_number_of_features_to_select));
	        slam_settings->addParameter(TypedParameter<int   > ("fv", "features_visible", "number_of_features_to_keep_visible", &number_of_features_to_keep_visible, &default_number_of_features_to_keep_visible));
	        slam_settings->addParameter(TypedParameter<int   > ("mif", "max_init_features", "max_features_to_init_at_once", &max_features_to_init_at_once, &default_max_features_to_init_at_once));
	        slam_settings->addParameter(TypedParameter<double> ("minl", "min_lambda", "min_lambda", &min_lambda, &default_min_lambda));
	        slam_settings->addParameter(TypedParameter<double> ("maxl", "max_lambda", "max_lambda", &max_lambda, &default_max_lambda));
	        slam_settings->addParameter(TypedParameter<int   > ("p", "particles", "number_of_particles", &number_of_particles, &default_number_of_particles));
	        slam_settings->addParameter(TypedParameter<double> ("dr", "deviation_ratio", "standard_deviation_depth_ratio", &standard_deviation_depth_ratio, &default_standard_deviation_depth_ratio));
	        slam_settings->addParameter(TypedParameter<int   > ("minp", "min_particles", "min_number_of_particles", &min_number_of_particles, &default_min_number_of_particles));
	        slam_settings->addParameter(TypedParameter<double> ("pt", "prune_threshold", "prune_probability_threshold", &prune_probability_threshold, &default_prune_probability_threshold));
	        slam_settings->addParameter(TypedParameter<int   > ("eia", "erase_init_after", "erase_partially_init_feature_after_this_many_attempts", &erase_partially_init_feature_after_this_many_attempts, &default_erase_partially_init_feature_after_this_many_attempts));
	        slam_settings->addParameter(TypedParameter<double> ("ckd", "cam_kd1", "cam_kd1", &cam_kd1, &default_cam_kd1));
	        slam_settings->addParameter(TypedParameter<int   > ("csd", "cam_sd", "cam_sd", &cam_sd, &default_cam_sd));
	        slam_settings->addParameter(TypedParameter<int   > ("ma", "minimum_attempted", "minimum_attempted", &minimum_attempted_measurements_of_feature, &default_minimum_attempted_measurements_of_feature));
	        slam_settings->addParameter(TypedParameter<double> ("smf", "successful_match_fraction", "successful_match_fraction", &successful_match_fraction, &default_successful_match_fraction));


    return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings) {


    //  =========  BASIC BUFFERS  (input / output )  =========

    g_monoslam = new SceneLib2::MonoSLAM ();

    init(g_monoslam, slam_settings);
	
    inputGrey = new cv::Mat ( grey_sensor->Height ,  grey_sensor->Width, CV_8UC1);

	pose_output = new slambench::outputs::Output("Position", slambench::values::VT_POSE, true);
	frame_output = new slambench::outputs::Output("Frame", slambench::values::VT_FRAME);
	frame_info_output = new slambench::outputs::Output("Frame (Annotated)", slambench::values::VT_FRAME);
	features_output = new slambench::outputs::Output("Features", slambench::values::VT_LIST);
	tracked_features_output = new slambench::outputs::Output("Tracked Features", slambench::values::VT_STRING);
	init_features_output = new slambench::outputs::Output("Init'd Features", slambench::values::VT_STRING);
	
	features_output->SetKeepOnlyMostRecent(true);
	frame_output->SetKeepOnlyMostRecent(true);
	frame_info_output->SetKeepOnlyMostRecent(true);
	tracked_features_output->SetKeepOnlyMostRecent(true);
	init_features_output->SetKeepOnlyMostRecent(true);
	
	slam_settings->GetOutputManager().RegisterOutput(pose_output);
	slam_settings->GetOutputManager().RegisterOutput(frame_output);
	slam_settings->GetOutputManager().RegisterOutput(frame_info_output);
	slam_settings->GetOutputManager().RegisterOutput(features_output);
	slam_settings->GetOutputManager().RegisterOutput(tracked_features_output);
	slam_settings->GetOutputManager().RegisterOutput(init_features_output);
	
	pose_output->SetActive(true);
	frame_output->SetActive(true);
	
    return true;
}

bool sb_update_frame (SLAMBenchLibraryHelper * slam_settings, slambench::io::SLAMFrame* s) {
	(void)slam_settings;
	
	if(s->FrameSensor == grey_sensor) {
		memcpy(inputGrey->data, s->GetData(), s->GetSize());
		s->FreeData();
		return true;
	}
	
	return false;
}

bool sb_process_once (SLAMBenchLibraryHelper * slam_settings) {
	(void)slam_settings;
	
	static bool first_frame = true;
	if(first_frame) {
		int attempts = 10;
		while((g_monoslam->feature_list_.size() < 4) && (attempts > 0)) {
			g_monoslam->InitialiseAutoFeature(*inputGrey);
			attempts--;
		}
//		g_monoslam->InitialiseAutoFeature(*inputGrey);
//		g_monoslam->InitialiseAutoFeature(*inputGrey);
//		g_monoslam->InitialiseAutoFeature(*inputGrey);
		first_frame = false;
	}
	
     g_monoslam->GoOneStep(*inputGrey,
             false,
             true);

     return true;

}


bool  sb_get_pose     (Eigen::Matrix4f * mat) {
    * mat =  g_monoslam->getPose();
    return true;
}
bool       sb_get_tracked  (bool * tracked) {
    *tracked = g_monoslam->number_of_visible_features_ > 0;
    return true;
}

bool sb_clean_slam_system(){
    return true;
}

static slambench::values::FrameValue *GetInfoFrame();

bool sb_update_outputs(SLAMBenchLibraryHelper* lib, const slambench::TimeStamp* latest_output) {
	(void)lib;
	(void)latest_output;
	lib->GetOutputManager().GetLock().lock();
	
	pose_output->AddPoint(*latest_output, new slambench::values::PoseValue(g_monoslam->getPose()));
	
	if(frame_output->IsActive()) {
		frame_output->AddPoint(*latest_output, new slambench::values::FrameValue(grey_sensor->Width, grey_sensor->Height, grey_sensor->PixelFormat, inputGrey->data));
	}
	
	if(features_output->IsActive()) {
		std::vector<slambench::values::Value*> features;
		for(SceneLib2::Feature *feature : g_monoslam->feature_list_) {
			slambench::values::FrameValue frame (g_monoslam->kBoxSize_, g_monoslam->kBoxSize_, slambench::io::pixelformat::G_I_8, feature->patch_.data);

			Eigen::Matrix4f pose;
			pose(0,1) = feature->y_(0);
			pose(0,2) = feature->y_(1);
			pose(0,3) = feature->y_(2);
			auto feature_value = new slambench::values::FeatureValue(pose, frame);
			features.push_back(feature_value);
		}
		features_output->AddPoint(*latest_output, new slambench::values::ValueListValue(features));
	}
	
	if(frame_info_output->IsActive()) {
		frame_info_output->AddPoint(*latest_output, GetInfoFrame());
	}
	
	if(tracked_features_output->IsActive()) {
		tracked_features_output->AddPoint(*latest_output, new slambench::values::TypeForVT<slambench::values::VT_STRING>::type(std::to_string(g_monoslam->feature_list_.size())));
	}
	if(init_features_output->IsActive()) {
		init_features_output->AddPoint(*latest_output, new slambench::values::TypeForVT<slambench::values::VT_STRING>::type(std::to_string(g_monoslam->feature_init_info_vector_.size())));
	}
	
	lib->GetOutputManager().GetLock().unlock();	
	return true;
}


struct RGBPixel {
public:
	RGBPixel(int R=0, int G=0, int B=0) : R(R), G(G), B(B) {
	}
	char R, G, B;
} __attribute__((packed));

class RGBImage {
public:
	
	RGBImage(int size_x, int size_y) : size_x(size_x), size_y(size_y) {
		pixels.resize(size_x * size_y);
	}
	
	int size_x, size_y;
	std::vector<RGBPixel> pixels;
	
	RGBPixel &Pxl(int x, int y) {
		assert(x >= 0);
		assert(y >= 0);
		assert(x < size_x);
		assert(y < size_y);
		return pixels.at(x + (size_x * y));
	}
};

RGBImage *convert_greyscale_to_rgb(const void* input, int size_x, int size_y) {
	const char *input_ptr = (char*)input;
	const char *input_end = ((char*)input) + size_x * size_y;
	
	RGBImage *image = new RGBImage (size_x, size_y);
	int count = 0;
	while(input_ptr != input_end) {
		image->pixels[count++] = RGBPixel(*input_ptr,*input_ptr,*input_ptr);
		input_ptr++;
	}
	
	return image;
}

void draw_rgb_box(RGBImage *image, int start_x, int start_y, int end_x, int end_y, RGBPixel colour) {
	if(start_x >= image->size_x) {
		return;
	}
	if(start_y >= image->size_y) {
		return;
	}
	if(end_x >= image->size_x) {
		return;
	}
	if(end_y >= image->size_y) {
		return;
	}
	
	if(start_y >= 0) {
		//draw top
		for(int i = start_x; i != end_x; ++i) {
			if(i >= image->size_x) {
				break;
			}
			if(i < 0) {
				break;
			}
			image->Pxl(i, start_y) = colour;
		}
	}
	
	if(start_x >= 0) {
		//draw lhs
		for(int i = start_y; i != end_y; ++i) {
			if(i >= image->size_y) {
				break;
			}
			if(i < 0) {
				break;
			}
			image->Pxl(start_x, i) = colour;
		}
	}
	
	if(end_x >= 0) {
		//draw rhs
		for(int i = start_y; i != end_y; ++i) {
			if(i >= image->size_y) {
				break;
			}
			if(i < 0) {
				break;
			}
			image->Pxl(end_x, i) = colour;
		}
	}
	
	if(end_y >= 0) {
		//draw bottom
		for(int i = start_x; i != end_x; ++i) {
			if(i >= image->size_x) {
				break;
			}
			if(i < 0) {
				break;
			}
			image->Pxl(i, end_y) = colour;
		}
	}
}

void draw_rgb_box(RGBImage *image, int start_x, int start_y, int end_x, int end_y, int width, RGBPixel colour) {
	if(width == 0) {
		return;
	}
	
	draw_rgb_box(image, start_x, start_y, end_x, end_y, colour);
	draw_rgb_box(image, start_x+1, start_y+1, end_x-1, end_y-1, width-1, colour);
}

static RGBImage *frame_data = nullptr;

static slambench::values::FrameValue *GetInfoFrame() 
{
	slambench::values::FrameValue *frame = new slambench::values::FrameValue(inputGrey->size[1], inputGrey->size[0], slambench::io::pixelformat::RGB_III_888);
	
	auto frame_data = convert_greyscale_to_rgb(inputGrey->data, inputGrey->size[1], inputGrey->size[0]);
	draw_rgb_box(frame_data, g_monoslam->init_feature_search_ustart_, g_monoslam->init_feature_search_vstart_, g_monoslam->init_feature_search_ufinish_, g_monoslam->init_feature_search_vfinish_, 4, RGBPixel(255,255,0));
	
	for(const auto *i : g_monoslam->feature_list_) {
		draw_rgb_box(frame_data, i->h_(0)-5, i->h_(1)-5, i->h_(0)+5, i->h_(1)+5, 2, RGBPixel(0,0,255));
	}
	
	memcpy(frame->GetData(), frame_data->pixels.data(), frame_data->pixels.size() * 3);
	
	delete frame_data;
	
	return frame;
}



