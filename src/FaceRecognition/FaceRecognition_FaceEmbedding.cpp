#include "FaceRecognition_FaceEmbedding.h"
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
using namespace dlib;
using namespace std; 
  
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

typedef struct Memory_ {
    int i;  
    anet_type net; 
} Memory;  

JNIEXPORT jlong JNICALL Java_FaceRecognition_FaceEmbedding_allocateMemory
  (JNIEnv *env, jobject, jstring pathToDnnData) {
    Memory* mem = new Memory();
    const char* cPathToDnnData = env->GetStringUTFChars(pathToDnnData, NULL);
    if (cPathToDnnData == NULL) {
      std::cerr<<"Cant find: " << cPathToDnnData<<std::endl;
		  delete mem;
		  return 0;
	  }
    mem->i = 0;
    std::cout<<mem->i<<std::endl;
    deserialize(cPathToDnnData) >> mem->net;
    return (jlong)mem;
}

JNIEXPORT void JNICALL Java_FaceRecognition_FaceEmbedding_detect
  (JNIEnv *env, jobject, jbyteArray imageData, jint width, jint height, jlong ptr) {
      Memory* mem = (Memory*)ptr;
      jint len = env->GetArrayLength(imageData);
      bool isRgb = len == 3 * width * height;
      std::cout<<mem->i<<" ohyeee " << len << " | isRgb: " <<isRgb<<" | "<<3 * width * height<<" vs"<<len<<std::endl;
      char *data = (char*)env->GetPrimitiveArrayCritical(imageData, 0);
      
      matrix<rgb_pixel> mat(1, len); 
      for (int i = 0; i < len; i+=3) {
        rgb_pixel p(data[i], data[i+1], data[i+2]); 
        mat(0, i)=rgb_pixel;
      }
      
      env->ReleasePrimitiveArrayCritical(imageData, data, 0);
}

JNIEXPORT void JNICALL Java_FaceRecognition_FaceEmbedding_freeMemory
  (JNIEnv *, jobject, jlong pointer) {
    delete (Memory*)pointer;
  }