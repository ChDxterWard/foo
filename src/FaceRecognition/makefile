FLAGS_COMPILE=-c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux
FLAGS_LINKING=-L/usr/local/lib -ldlib -pthread -lX11 -shared

faceEmbedding: FaceRecognition_FaceEmbedding.o MyImage.o
	g++ MyImage.o FaceRecognition_FaceEmbedding.o  -lopencv_core -lopencv_imgcodecs -lopencv_imgproc $(FLAGS_LINKING) -o libfaceEmbedding.so 

FaceRecognition_FaceEmbedding.o: FaceRecognition_FaceEmbedding.cpp FaceRecognition_FaceEmbedding.h
	g++  FaceRecognition_FaceEmbedding.cpp -I/usr/local/include/opencv4 $(FLAGS_COMPILE) -o FaceRecognition_FaceEmbedding.o

MyImage.o: MyImage.cpp MyImage.h
	g++ MyImage.cpp -I/usr/local/include/dlib/image_processing/ $(FLAGS_COMPILE) -o MyImage.o