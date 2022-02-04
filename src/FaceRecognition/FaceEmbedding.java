package FaceRecognition;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class FaceEmbedding {
	
	static {
        System.loadLibrary("faceEmbedding");
    }
	private static long mem;
	public static void main(String[] args) {
		mem = new FaceEmbedding().allocateMemory("/home/r/eclipse-workspace/foo/src/FaceRecognition/dlib_face_recognition_resnet_model_v1.dat");
		
		BufferedImage bufferedImage = bar("/home/r/Bilder/face.png");
		BufferedImage e1 = bar("/home/r/Bilder/e1.png");
		BufferedImage e2 = bar("/home/r/Bilder/e2.png");

		byte[] imageData1 = foo(bufferedImage);
		byte[] imageData2 = foo(e1);
		byte[] imageData3 = foo(e2);
		float[] ret1 =new FaceEmbedding().encode(imageData1, bufferedImage.getWidth(), bufferedImage.getHeight(), mem);
		float[] ret2 =new FaceEmbedding().encode(imageData2, bufferedImage.getWidth(), bufferedImage.getHeight(), mem);
		float[] ret3 =new FaceEmbedding().encode(imageData3, bufferedImage.getWidth(), bufferedImage.getHeight(), mem);
		System.out.println("e1 vs e2 " + calcDist(ret2, ret3));
		System.out.println("e2 vs e1 " + calcDist(ret2, ret3));
		System.out.println("e0 vs e1 " + calcDist(ret1, ret2));
		System.out.println("e0 vs e2 " + calcDist(ret1, ret3));
	}
	static double calcDist(float[] a1, float[] a2) {
		assert a1.length == a2.length;
		double dist = .0;
		for (int i = 0; i < a1.length; i++) {
			dist += Math.pow(a1[i] - a2[i], 2);
		}
		return Math.sqrt(dist);
	}
	public static BufferedImage bar(String url) {
		try {
			BufferedImage img = ImageIO.read(new File(url));
			return img;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}
	public static byte[] foo(BufferedImage img) {
		
		
//		      ByteArrayOutputStream baos = new ByteArrayOutputStream();
//		      try {
//				ImageIO.write(img, "png", baos);
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//		      byte[] bytes = baos.toByteArray();
//		      return bytes;

//		int colStride = 3;
//		int rowStride = colStride * img.getWidth();
//		byte[] values = new byte[img.getHeight() * img.getWidth() * 3];
//		DataBuffer colorDataBuffer = img.getRaster().getDataBuffer();
//		byte[] byteValues = new byte[3];
//		for (int i = 0; i < img.getWidth() * img.getHeight(); i++) {
//			 intToByteArray(colorDataBuffer.getElem(i), byteValues);
//			for (int j = 0; j < 3; j++) {
//				values[(i * colStride) + j] = byteValues[j];
//			} 
//		}
//		return values;
		byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
		return pixels;
	}
	private static void intToByteArray(int value, byte[] values) {
		if(values.length != 3)
			throw new IllegalArgumentException("Given array must have length of three!");
		// mask for each channel
		int redMask = 0x00FF0000;
		int greenMask = 0x0000FF00;
		int blueMask = 0x000000FF;
		values[0] =  (byte) ((value & redMask) >> 16);
		values[1] = (byte) ((value & greenMask) >> 8);
		values[2] = (byte) (value & blueMask); 
	}
	

	private native long allocateMemory(String pathToDnnData);
	private native float[] encode(byte[] imageData, int width, int height, long pointer);
	private native void freeMemory(long pointer);
}
