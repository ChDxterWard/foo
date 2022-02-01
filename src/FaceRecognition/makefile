FLAGS_COMPILE=-c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux
FLAGS_LINKING=-L/usr/local/lib -ldlib -pthread -lX11 -shared

faceEmbedding: FaceRecognition_FaceEmbedding.o
	g++ FaceRecognition_FaceEmbedding.o  $(FLAGS_LINKING) -o libfaceEmbedding.so 

FaceRecognition_FaceEmbedding.o: MyImage.o FaceRecognition_FaceEmbedding.cpp FaceRecognition_FaceEmbedding.h
	g++  FaceRecognition_FaceEmbedding.cpp MyImage.o $(FLAGS_COMPILE) -o FaceRecognition_FaceEmbedding.o

MyImage.o: MyImage.cpp
	g++ MyImage.cpp -o MyImage.o