//#include "generic_image.h"
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
using namespace dlib;
using namespace std; 

namespace dlib
    {
        template <> 
        struct image_traits<MyImage>
        {
            typedef rgb_pixel pixel_type;
        };

        class MyImage {       
  public:             
    int myNum;   
    string myString;
    
};
    }

