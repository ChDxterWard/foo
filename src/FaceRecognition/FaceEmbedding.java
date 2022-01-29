package FaceRecognition;

import java.awt.image.BufferedImage;
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
		
		BufferedImage bufferedImage = bar();
		byte[] imageData = foo(bufferedImage);
		new FaceEmbedding().detect(imageData, bufferedImage.getWidth(), bufferedImage.getHeight(), mem);
	}
	public static BufferedImage bar() {
		try {
			BufferedImage img = ImageIO.read(new File("/home/r/Bilder/face.png"));
			return img;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}
	public static byte[] foo(BufferedImage img) {
		
		
		      ByteArrayOutputStream baos = new ByteArrayOutputStream();
		      try {
				ImageIO.write(img, "png", baos);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		      byte[] bytes = baos.toByteArray();
		      return bytes;

	}
	

	private native long allocateMemory(String pathToDnnData);
	private native void detect(byte[] imageData, int width, int height, long pointer);
	private native void freeMemory(long pointer);
}
