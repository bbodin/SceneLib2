/*
 * cv_mat.h
 *
 *  Created on: 1 Jul 2016
 *      Author: toky
 */

#include "core/core.hpp"
#define BENCHMARKS_MONOSLAM_SRC_SEQUENTIAL_SCENELIB2_CV_MAT_H_
#ifndef BENCHMARKS_MONOSLAM_SRC_SEQUENTIAL_SCENELIB2_CV_MAT_H_
#define BENCHMARKS_MONOSLAM_SRC_SEQUENTIAL_SCENELIB2_CV_MAT_H_

#include <iostream>
#include <cassert>
#include <vector>

#define CV_64FC1 1
#define CV_8UC1  2


namespace cv {

class cv_size {
    public :
    int width;
    int height;
};

class Mat {

private :
    int _type = -1;
    cv_size _size = {0,0};
    std::vector<unsigned char> _data;

public :
    unsigned char * data = NULL;

private :
    void update(double scalar) {
    	   if (this->_type == CV_8UC1) {
    	        	_data.resize(this->_size.width*this->_size.height);
    	            this->data = (unsigned char *) &(this->_data[0]);
    	            for (int i = 0 ; i < this->_size.height * this->_size.width ; i++) {
    	                    this->data[i] = scalar;
    	                }
    	        } else if (this->_type == CV_64FC1) {
    	        	_data.resize(this->_size.width*this->_size.height * (sizeof(double) / sizeof(char)));
    	            this->data = (unsigned char *) &(this->_data[0]);
    	            for (int i = 0 ; i < this->_size.height * this->_size.width ; i++) {
    	                    *((double*)(this->data) + i) = scalar;
    	                }
    	        } else {
    	            std::cerr << "Type invalid = " << this->_type << std::endl;
    	            assert (this->_type == CV_64FC1 || this->_type == CV_8UC1);
    	        }

    }

    public :

    Mat() {};
    //cv_mat(const cv_mat&) {assert(false);};


    Mat(cv_size hw, int t, double scalar)  {

        this->_size = hw;
        this->_type = t;

        this->update(scalar);




    }


    Mat(int rows, int cols, int t)  {


        this->_size.width = cols;
        this->_size.height = rows;
        this->_type = t;

        this->update(0);

    }

    ~Mat () {

    }

    const cv_size size() const {
        return this->_size;
    }

    template <typename _Tp>
    _Tp& at( unsigned int row , unsigned int col) const {
        return ((_Tp*)(data + row * this->size().width))[col];
    }

    int type() const {
        return _type;
    }

    bool empty() {
        return ((_size.height * _size.width) == 0);
    }

};

}

#endif /* BENCHMARKS_MONOSLAM_SRC_SEQUENTIAL_SCENELIB2_CV_MAT_H_ */
