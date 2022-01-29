# Create headers
#javac -h . test.java
# Write code in the *.cpp file
# Compile
##g++ -c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux foo_test.cpp -o foo_test.o
# Make shared object with name libnative.so. We have to load it in java as native.
##g++ -shared -fPIC -o libnative.so foo_test.o -lc

# Compile dnn_face_recognition_ex.cpp
g++ dnn_face_recognition_ex.cpp -ldlib -pthread -lX11 -o a.out
#a.out faces/bald_guys.jpg 

#Wichtig in cmake fuer dlib im Bereich CMAKE_CXX_FLAGS -fPIC angeben!!!

g++ -c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux foo_test.cpp -ldlib -pthread -lX11  -o foo_test.o
g++ -shared -fPIC -ldlib -pthread -lX11 -o libnative.so foo_test.o -lc