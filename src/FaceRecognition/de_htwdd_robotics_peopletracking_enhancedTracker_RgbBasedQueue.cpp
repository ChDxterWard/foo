// dependencies:
// Boost 1.48+
// OpenCV 2.4+
// https://github.com/ex-ratt/AdapTrack tag 1.0-ubuntu14 (installed in /opt/AdapTrack/install)

// compilation command: (assumes OpenCV to be installed in /usr - otherwise add -I and -L options that point to the directories)
// g++ -I/opt/AdapTrack/install/include -I/usr/lib/jvm/default-java/include -I/usr/lib/jvm/default-java/include/linux -Wall de_htwdd_robotics_peopletracking_particle_measurement_RgbBasedQueue.cpp -L/opt/AdapTrack/install/lib -lDetection -lSVM -lClassification -lImageProcessing -lopencv_core -std=c++11 -shared -fPIC -o libRgbBasedQueue.so

#include "de_htwdd_robotics_peopletracking_enhancedTracker_RgbBasedQueue.h"
#include "classification/IncrementalClassifierTrainer.hpp"
#include "classification/IncrementalLinearSvmTrainer.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/ProbabilisticSupportVectorMachine.hpp"
#include "classification/PseudoProbabilisticSvmTrainer.hpp"
#include "classification/SupportVectorMachine.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "detection/NonMaximumSuppression.hpp"
#include "imageprocessing/extraction/AggregatedFeaturesExtractor.hpp"
#include "imageprocessing/extraction/ExactFhogExtractor.hpp"
#include "imageprocessing/filtering/FhogFilter.hpp"
#include "libsvm/LibSvmTrainer.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <typeinfo>
//#include <opencv2/imgcodecs.hpp>

using classification::IncrementalClassifierTrainer;
using classification::IncrementalLinearSvmTrainer;
using classification::LinearKernel;
using classification::ProbabilisticSupportVectorMachine;
using classification::PseudoProbabilisticSvmTrainer;
using classification::SupportVectorMachine;
using detection::AggregatedFeaturesDetector;
using detection::NonMaximumSuppression;
using imageprocessing::Patch;
using imageprocessing::extraction::AggregatedFeaturesExtractor;
using imageprocessing::extraction::ExactFhogExtractor;
using imageprocessing::filtering::FhogFilter;
using libsvm::LibSvmTrainer;
using namespace cv;
using namespace std;

typedef struct Memory_ {
	shared_ptr<AggregatedFeaturesDetector> detector; ///< Head detector.
	shared_ptr<AggregatedFeaturesExtractor> extractor; ///< Feature extractor that uses a FHOG feature pyramid for fast extraction.
	shared_ptr<ExactFhogExtractor> exactExtractor; ///< Feature extractor that extracts FHOG features at the exact given position.
	shared_ptr<ProbabilisticSupportVectorMachine> probSvm; ///< Probabilistic SVM that computes the likelihood of a visible head existing given FHOG features.
	shared_ptr<IncrementalClassifierTrainer<ProbabilisticSupportVectorMachine>> probSvmTrainer; ///< Trainer of probabilistic SVMs given training data.
	unordered_set<jint> inactiveTrackIds; ///< Identifiers of tracks that do not exist anymore.
	unordered_map<jint, shared_ptr<ProbabilisticSupportVectorMachine>> trackSvms; ///< Target-specific probabilistic SVMs that are trained on-line.
	Mat image; ///< Current grayscale image.
	std::default_random_engine rng; ///< Random number generator.
	double visibilityThreshold; ///< SVM score that must be exceeded to consider a target visible.
	double probVisible; ///< Probability that a person's head is visible in the image.
	double probHeadInvisible; ///< Probability of the existence of a person if it is not visible.
	bool adaptive; ///< Flag that indicates whether on-line SVMs are trained and used or not.
	int negativeExampleCount; ///< Number of negative training examples for each SVM training.
	double negativeOverlapThreshold; ///< Maximum allowed overlap ratio between negative and positive training examples.
} Memory;
#include <unistd.h>
JNIEXPORT jlong JNICALL Java_de_htwdd_robotics_peopletracking_enhancedTracker_RgbBasedQueue_allocateMemory
  (JNIEnv* env, jobject, jstring svmFilename, jint cellSize, jint minSize, jdouble detectionThreshold, jdouble visibilityThreshold, jdouble probVisible, jdouble probHeadInvisible, jdouble learnRate) {
	Memory* mem = new Memory();
	//printf("OpenCV: %s", cv::getBuildInformation().c_str());
	// load SVM using svmFilename and detectionThreshold
	const char* cSvmFilename = env->GetStringUTFChars(svmFilename, NULL);
	if (cSvmFilename == NULL) { // OutOfMemoryError already thrown
		delete mem;
		return 0;
	}
	ifstream svmStream(cSvmFilename);
	mem->probSvm = ProbabilisticSupportVectorMachine::load(svmStream);
	svmStream.close();
	env->ReleaseStringUTFChars(svmFilename, NULL);
	//env->ReleaseStringUTFChars(svmFilename, cSvmFilename);
	auto svm = mem->probSvm->getSvm();
	svm->setThreshold(static_cast<float>(detectionThreshold));
	mem->probSvm->setLogisticB(0.0); // shift logistic function to zero, so SVM score of zero equals a probability of 50% (logistic function parameter computation ignored training data weights)
	// check SVM
	if (svm->getSupportVectors().size() != 1) {
		cerr << "Invalid SVM: must have exactly one support vector, but had: " << svm->getSupportVectors().size() << endl;
		delete mem;
		return 0;
	}
	int histogramChannelCount = svm->getSupportVectors()[0].channels() - 4;
	if (histogramChannelCount % 3 != 0) {
		cerr << "Invalid SVM: depth must be 3n+4, but was: " << svm->getSupportVectors()[0].channels() << endl;
		delete mem;
		return 0;
	}
	int windowWidth = svm->getSupportVectors()[0].cols;
	int windowHeight = svm->getSupportVectors()[0].rows;
	if (windowWidth != windowHeight) {
		cerr << "Invalid SVM: feature window must be square, but was: " << windowWidth << 'x' << windowHeight << endl;
		delete mem;
		return 0;
	}
	int binCount = histogramChannelCount / 3;
	// create FHOG filter, non-maximum suppression, detector, and feature extractors
	shared_ptr<FhogFilter> fhogFilter = make_shared<FhogFilter>(cellSize, binCount, false, true, 0.2f);
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(
			0.3, NonMaximumSuppression::MaximumType::WEIGHTED_AVERAGE);
	mem->detector = make_shared<AggregatedFeaturesDetector>(fhogFilter, cellSize, Size(windowWidth, windowHeight), 5, svm, nms, 1.0, 1.0, minSize);
	mem->extractor = mem->detector->getFeatureExtractor();
	mem->exactExtractor = make_shared<ExactFhogExtractor>(fhogFilter, windowWidth, windowHeight);
	// create probabilistic SVM trainer
	auto libSvmTrainer = make_shared<LibSvmTrainer>(10, true);
	auto incrementalSvmTrainer = make_shared<IncrementalLinearSvmTrainer>(libSvmTrainer, learnRate);
	mem->probSvmTrainer = make_shared<PseudoProbabilisticSvmTrainer>(incrementalSvmTrainer, 0.95, 0.05, 1.0, -1.0);

	// initialize random number generator
	mem->rng.seed(random_device()());

	// set parameters
	mem->visibilityThreshold = visibilityThreshold;
	mem->probVisible = probVisible;
	mem->probHeadInvisible = probHeadInvisible;
	mem->adaptive = learnRate > 0.0;
	mem->negativeExampleCount = 10;
	mem->negativeOverlapThreshold = 0.5;
	return (jlong)mem;
}

JNIEXPORT jshortArray JNICALL Java_de_htwdd_robotics_peopletracking_enhancedTracker_RgbBasedQueue_detect
  (JNIEnv* env, jobject, jbyteArray imageData, jint width, jint height, jlong pointer) {
	if (pointer == 0) {
		cerr << "Invalid memory pointer" << endl;
		return NULL;
	}
	Memory* mem = (Memory*)pointer;

	// remove SVMs of inactive tracks
	for (jint trackId : mem->inactiveTrackIds)
		mem->trackSvms.erase(trackId);

	// reset inactive track IDs
	mem->inactiveTrackIds.clear();
	for (const auto& trackEntry : mem->trackSvms)
		mem->inactiveTrackIds.insert(trackEntry.first);

	// get image
	uchar* data = (uchar*)env->GetPrimitiveArrayCritical(imageData, 0);
	jsize dataLength = env->GetArrayLength(imageData);
	bool isRgb = dataLength == 3 * width * height;
	if (isRgb) {
		Mat image(height, width, CV_8UC3, data);
		cvtColor(image, mem->image, CV_RGB2GRAY);
	} else {
		Mat image(height, width, CV_8UC1, data);
		image.copyTo(mem->image);
	}
	env->ReleasePrimitiveArrayCritical(imageData, data, 0);

	// detect heads
	vector<Rect> detections = mem->detector->detect(mem->image);
	mem->exactExtractor->update(mem->image);
	jshort detectionData[3 * detections.size()];
	for (size_t i = 0; i < detections.size(); ++i) {
		detectionData[3 * i] = detections[i].x;
		detectionData[3 * i + 1] = detections[i].y;
		detectionData[3 * i + 2] = detections[i].height;
	}
	jshortArray detectionDataArray = env->NewShortArray(3 * detections.size());
	if (detectionDataArray == NULL) // OutOfMemoryError already thrown
		return NULL;
	env->SetShortArrayRegion(detectionDataArray, 0, 3 * detections.size(), detectionData);
	
	return detectionDataArray;
}

JNIEXPORT void JNICALL Java_de_htwdd_robotics_peopletracking_enhancedTracker_RgbBasedQueue_getLikelihoods
  (JNIEnv* env, jobject, jint trackId, jshortArray particleData, jfloatArray likelihoodData, jlong pointer) {
	if (pointer == 0) {
		cerr << "Invalid memory pointer" << endl;
		return;
	}
	Memory* mem = (Memory*)pointer;
		// remove ID from inactive track IDs
	mem->inactiveTrackIds.erase(trackId);

	// compute likelihoods of head existence of the given square bounding boxes
	jshort* data = (jshort*)env->GetPrimitiveArrayCritical(particleData, 0);
	jfloat* likelihoods = (jfloat*)env->GetPrimitiveArrayCritical(likelihoodData, 0);
	jsize count = env->GetArrayLength(likelihoodData);
	for (jsize i = 0; i < count; ++i) {
		shared_ptr<Patch> featurePatch = mem->extractor->extract(
				Rect(data[3 * i], data[3 * i + 1], data[3 * i + 2], data[3 * i + 2]));
		double probHeadVisible = 0;
		if (featurePatch) {
			probHeadVisible = mem->probSvm->getProbability(featurePatch->getData()).second;
			if (mem->trackSvms.count(trackId) == 1)
				probHeadVisible = sqrt(probHeadVisible * mem->trackSvms[trackId]->getProbability(featurePatch->getData()).second);
		}
		likelihoods[i] = mem->probVisible * probHeadVisible + (1 - mem->probVisible) * mem->probHeadInvisible;
		//std::cout<<probHeadVisible<< " | " << likelihoods[i]<<std::endl;
	}
	env->ReleasePrimitiveArrayCritical(likelihoodData, likelihoods, 0);
	env->ReleasePrimitiveArrayCritical(particleData, data, 0);
}

JNIEXPORT void JNICALL Java_de_htwdd_robotics_peopletracking_enhancedTracker_RgbBasedQueue_getScores
  (JNIEnv* env, jobject, jintArray trackIdData, jshortArray boxData, jfloatArray scoreData, jlong pointer) {
	if (pointer == 0) {
		cerr << "Invalid memory pointer" << endl;
		return;
	}
	Memory* mem = (Memory*)pointer;
	// retrieve SVM scores of the given tracks using exactly extracted FHOG features
	jint* trackIds = (jint*)env->GetPrimitiveArrayCritical(trackIdData, 0);
	jshort* boxes = (jshort*)env->GetPrimitiveArrayCritical(boxData, 0);
	jfloat* scores = (jfloat*)env->GetPrimitiveArrayCritical(scoreData, 0);
	jsize count = env->GetArrayLength(trackIdData);
	//std::cout<<"count: " << count<<std::endl;
	for (jsize i = 0; i < count; ++i) {
	//	std::cout<<"i: " << i<<std::endl;
		jint trackId = trackIds[i];
		Rect boundingBox(boxes[3 * i], boxes[3 * i + 1], boxes[3 * i + 2], boxes[3 * i + 2]);
		shared_ptr<Patch> featurePatch = nullptr;
				
			featurePatch = mem->exactExtractor->extract(boundingBox);
//std::cout<<"channels: "<<featurePatch->getData().channels()<<std::endl;
//cv::imwrite("~/work/foo.jpg",featurePatch->getData()); 

		if (!featurePatch) {
			scores[i] = -101;
		} else {


			//std::cout<<typeid(featurePatch->getData()).name()<< " | " <<boxes[3 * i]<<" | "<< boxes[3 * i + 1]<<" | "<<  boxes[3 * i + 2]<<std::endl;
			if (mem->adaptive) { // use on-line target-specific SVM
				if (mem->trackSvms.count(trackId) == 1)
					scores[i] = static_cast<float>(mem->trackSvms[trackId]->getSvm()->computeHyperplaneDistance(featurePatch->getData()));
				else
					scores[i] = -102;
			} else { // use off-line head SVM
				scores[i] = static_cast<float>(mem->probSvm->getSvm()->computeHyperplaneDistance(featurePatch->getData()));
	//std::cout<<"scores[i]: " << scores[i] << std::endl;
			}
		}
	}
	env->ReleasePrimitiveArrayCritical(scoreData, scores, 0);
	env->ReleasePrimitiveArrayCritical(boxData, boxes, 0);
	env->ReleasePrimitiveArrayCritical(trackIdData, trackIds, 0);
}

vector<Mat> getNegativeTrainingExamples(Memory* mem, Rect target);
vector<Mat> getNegativeTrainingExamples(Memory* mem, Rect target, const SupportVectorMachine& svm);
double computeOverlap(Rect a, Rect b);

JNIEXPORT void JNICALL Java_de_htwdd_robotics_peopletracking_enhancedTracker_RgbBasedQueue_update
  (JNIEnv* env, jobject, jintArray trackIdData, jshortArray boxData, jlong pointer) {
	if (pointer == 0) {
		cerr << "Invalid memory pointer" << endl;
		return;
	}
	Memory* mem = (Memory*)pointer;

	if (!mem->adaptive)
		return;

	// update target-specific SVMs of the tracks
	jint* trackIds = (jint*)env->GetPrimitiveArrayCritical(trackIdData, 0);
	jshort* boxes = (jshort*)env->GetPrimitiveArrayCritical(boxData, 0);
	jsize count = env->GetArrayLength(trackIdData);
	for (jsize i = 0; i < count; ++i) {
		jint trackId = trackIds[i];
		Rect boundingBox(boxes[3 * i], boxes[3 * i + 1], boxes[3 * i + 2], boxes[3 * i + 2]);
		shared_ptr<Patch> featurePatch = mem->exactExtractor->extract(boundingBox);
		if (featurePatch
				&& boundingBox.x >= 0
				&& boundingBox.y >= 0
				&& boundingBox.x + boundingBox.width <= mem->image.cols
				&& boundingBox.y + boundingBox.height <= mem->image.rows) {
			bool trackSvmExists = mem->trackSvms.count(trackId) == 1;
			if (trackSvmExists) {
				mem->probSvmTrainer->retrain(*mem->trackSvms[trackId], vector<Mat>{featurePatch->getData()}, getNegativeTrainingExamples(mem, boundingBox, *mem->trackSvms[trackId]->getSvm()));
			} else {
				auto probabilisticSvm = make_shared<ProbabilisticSupportVectorMachine>(make_shared<LinearKernel>());
				mem->probSvmTrainer->train(*probabilisticSvm, vector<Mat>{featurePatch->getData()}, getNegativeTrainingExamples(mem, boundingBox));
				mem->trackSvms.emplace(trackId, probabilisticSvm);
			}
		}
	}
	env->ReleasePrimitiveArrayCritical(boxData, boxes, 0);
	env->ReleasePrimitiveArrayCritical(trackIdData, trackIds, 0);
}

vector<Mat> getNegativeTrainingExamples(Memory* mem, Rect target) {
	int lowerX = target.x - target.width;
	int upperX = target.x + target.width;
	int lowerY = target.y - target.height;
	int upperY = target.y + target.height;
	int lowerH = target.height / 2;
	int upperH = target.height * 2;
	vector<Mat> trainingExamples;
	trainingExamples.reserve(mem->negativeExampleCount);
	while (trainingExamples.size() < trainingExamples.capacity()) {
		int x = uniform_int_distribution<int>{lowerX, upperX}(mem->rng);
		int y = uniform_int_distribution<int>{lowerY, upperY}(mem->rng);
		int height = uniform_int_distribution<int>{lowerH, upperH}(mem->rng);
		int width = height * target.width / target.height;
		shared_ptr<Patch> patch = mem->extractor->extract(Rect(x, y, width, height));
		if (patch && computeOverlap(target, patch->getBounds()) <= mem->negativeOverlapThreshold)
			trainingExamples.push_back(patch->getData());
	}
	return trainingExamples;
}

vector<Mat> getNegativeTrainingExamples(Memory* mem, Rect target, const SupportVectorMachine& svm) {
	int lowerX = target.x - target.width;
	int upperX = target.x + target.width;
	int lowerY = target.y - target.height;
	int upperY = target.y + target.height;
	int lowerH = target.height / 2;
	int upperH = target.height * 2;
	vector<pair<double, Mat>> trainingCandidates;
	trainingCandidates.reserve(3 * mem->negativeExampleCount);
	while (trainingCandidates.size() < trainingCandidates.capacity()) {
		int x = uniform_int_distribution<int>{lowerX, upperX}(mem->rng);
		int y = uniform_int_distribution<int>{lowerY, upperY}(mem->rng);
		int height = uniform_int_distribution<int>{lowerH, upperH}(mem->rng);
		int width = height * target.width / target.height;
		shared_ptr<Patch> patch = mem->extractor->extract(Rect(x, y, width, height));
		if (patch && computeOverlap(target, patch->getBounds()) <= mem->negativeOverlapThreshold) {
			double score = svm.computeHyperplaneDistance(patch->getData());
			trainingCandidates.emplace_back(score, patch->getData());
		}
	}
	partial_sort(trainingCandidates.begin(), trainingCandidates.begin() + mem->negativeExampleCount, trainingCandidates.end(),
			[](const pair<double, Mat>& a, const pair<double, Mat>& b) { return a.first > b.first; });
	vector<Mat> trainingExamples;
	trainingExamples.reserve(mem->negativeExampleCount);
	for (size_t i = 0; i < trainingExamples.capacity(); ++i)
		trainingExamples.push_back(trainingCandidates[i].second);
	return trainingExamples;
}

double computeOverlap(Rect a, Rect b) {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

JNIEXPORT void JNICALL Java_de_htwdd_robotics_peopletracking_enhancedTracker_RgbBasedQueue_freeMemory
  (JNIEnv*, jobject, jlong pointer) {
	delete (Memory*)pointer;
}

