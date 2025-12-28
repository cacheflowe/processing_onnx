
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;

import processing.video.Capture;

// ONNX Runtime setup
OrtEnvironment env;
OrtSession session;
ArrayList<Person> detectedPeople = new ArrayList<Person>();
PImage inputResized;

// Reusable buffers for inference (avoid allocations each frame)
IntBuffer inputBuffer;
String inputName;
long[] inputShape;

// Threading for async inference
ExecutorService inferenceExecutor;
AtomicBoolean inferenceRunning = new AtomicBoolean(false);
volatile ArrayList<Person> pendingResults = new ArrayList<Person>();
float lastMinConfidence = 0.3f;
boolean lastFlipY = false;

// Webcam
PImage testImg;
Capture webcam;

// UI controls
float minConfidence = 0.3f;
boolean flipY = false;

// Skeleton connections
int[][] CONNECTIONS = {
  {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 7}, {7, 9},
  {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

void settings() {
  size(1280, 720);
}

void setup() {
  loadImage();
  // loadWebcam();
  initializeONNX();
}

void loadImage() {
  testImg = loadImage("zachary-nelson-98Elr-LIvD8-unsplash.jpg");
}

void loadWebcam() {
  // Initialize webcam
  String[] cameras = Capture.list();
  if (cameras.length == 0) {
    println("No cameras available!");
    exit();
  } else {
    println("Available cameras:");
    for (int i = 0; i < cameras.length; i++) {
      println(i + ": " + cameras[i]);
    }
    webcam = new Capture(this, 640, 480);
    webcam.start();
  }
}

PImage getInputImage() {
  if (webcam != null && webcam.available()) {
    webcam.read();
    return webcam;
  } else {
    return testImg;
  }
}

void initializeONNX() {
  // Init ONNX session
  try {
    println("Available ONNX providers: " + OrtEnvironment.getAvailableProviders());
    env = OrtEnvironment.getEnvironment();
    
    SessionOptions sessionOptions = new SessionOptions();
    
    boolean useCUDA = false;
    if (useCUDA) {
      try {
        sessionOptions.addCUDA(0);
        println("CUDA execution provider added successfully");
      } catch (OrtException e) {
        println("CUDA not available, falling back to CPU: " + e.getMessage());
      }
    } else {
      println("Running on CPU (CUDA disabled)");
    }
    
    // Load model from data folder
    String modelPath = dataPath("movenet-multipose-lightning.onnx");
    println("Loading model from: " + modelPath);
    session = env.createSession(modelPath, sessionOptions);
    
    println("Input info: " + session.getInputInfo());
    println("Output info: " + session.getOutputInfo());
    
    inputName = session.getInputNames().iterator().next();
  } catch (OrtException e) {
    e.printStackTrace();
  }
  
  // Pre-allocate input buffer (256x256x3 RGB as INT32)
  int bufferSize = 256 * 256 * 3;
  inputBuffer = ByteBuffer.allocateDirect(bufferSize * 4)
    .order(ByteOrder.nativeOrder())
    .asIntBuffer();
  inputShape = new long[]{1, 256, 256, 3};
  
  inputResized = createImage(256, 256, ARGB);
  
  // Create single-thread executor for inference
  inferenceExecutor = Executors.newSingleThreadExecutor();
}

void draw() {
  background(0);  
  PImage inputImage = getInputImage();
  
  // Run inference on latest frame (async)
  if (session != null && inputImage != null && inputImage.width > 10) {
    // Copy source image to input buffer, resizing to 256x256
    inputResized.copy(inputImage, 0, 0, inputImage.width, inputImage.height, 0, 0, 256, 256);
    image(inputImage, 0, 0, width, height);
    
    // Only submit new inference if previous one is done
    if (!inferenceRunning.get()) {
      submitInference(inputResized);
    }
    
    // Copy pending results to detected people (lock-free swap)
    ArrayList<Person> latest = pendingResults;
    if (latest != null && latest != detectedPeople) {
      detectedPeople = latest;
    }
    
    drawSkeletons(width, height);
  }
  
  // Draw UI
  drawUI();
}

void drawUI() {
  fill(0, 150);
  noStroke();
  rectMode(CORNER);
  rect(0, height - 80, 300, 80);
  
  fill(255);
  textSize(12);
  text("Min Confidence: " + nf(minConfidence, 1, 2) + " (UP/DOWN arrows)", 20, height - 55);
  text("Flip Y: " + flipY + " (press 'F' to toggle)", 20, height - 35);
  text("FPS: " + nf(frameRate, 2, 1), 20, height - 15);
}

void keyPressed() {
  if (key == 'f' || key == 'F') {
    flipY = !flipY;
  }
  if (keyCode == UP) {
    minConfidence = constrain(minConfidence + 0.05, 0, 1);
  }
  if (keyCode == DOWN) {
    minConfidence = constrain(minConfidence - 0.05, 0, 1);
  }
}

void submitInference(PImage img) {
  img.loadPixels();
  // Clone pixels - async thread needs its own copy
  final int[] pixelsCopy = img.pixels.clone();
  final int imgWidth = img.width;
  final int imgHeight = img.height;
  final float capturedMinConf = minConfidence;
  final boolean capturedFlipY = flipY;
  
  inferenceRunning.set(true);
  inferenceExecutor.submit(new Runnable() {
    public void run() {
      try {
        runMovenetAsync(pixelsCopy, imgWidth, imgHeight, capturedFlipY, capturedMinConf);
      } finally {
        inferenceRunning.set(false);
      }
    }
  });
}

void runMovenetAsync(int[] pixels, int imgWidth, int imgHeight, boolean flipY, float minConf) {
  try {
    inputBuffer.clear();
    
    if (!flipY) {
      for (int i = 0; i < pixels.length; i++) {
        int pixel = pixels[i];
        inputBuffer.put((pixel >> 16) & 0xFF); // R
        inputBuffer.put((pixel >> 8) & 0xFF);  // G
        inputBuffer.put(pixel & 0xFF);         // B
      }
    } else {
      for (int y = imgHeight - 1; y >= 0; y--) {
        for (int x = 0; x < imgWidth; x++) {
          int pixel = pixels[y * imgWidth + x];
          inputBuffer.put((pixel >> 16) & 0xFF);
          inputBuffer.put((pixel >> 8) & 0xFF);
          inputBuffer.put(pixel & 0xFF);
        }
      }
    }
    inputBuffer.rewind();
    
    OnnxTensor tensor = OnnxTensor.createTensor(env, inputBuffer, inputShape);
    Result result = session.run(Collections.singletonMap(inputName, tensor));
    
    float[][][] output = (float[][][]) result.get(0).getValue();
    float[][] people = output[0];
    
    ArrayList<Person> newResults = new ArrayList<Person>();
    
    for (int i = 0; i < people.length; i++) {
      float[] personData = people[i];
      float confidence = personData[55];
      if (confidence >= minConf) {
        Person person = new Person();
        person.confidence = confidence;
        person.bbox[0] = personData[51];
        person.bbox[1] = personData[52];
        person.bbox[2] = personData[53];
        person.bbox[3] = personData[54];
        
        for (int k = 0; k < 17; k++) {
          float y = personData[k * 3];
          float x = personData[k * 3 + 1];
          float score = personData[k * 3 + 2];
          person.keypoints.add(new Keypoint(x, y, score));
        }
        newResults.add(person);
      }
    }
    
    // Atomic swap - create new list and assign (lock-free)
    pendingResults = newResults;
    
    result.close();
    tensor.close();
    
  } catch (OrtException e) {
    e.printStackTrace();
  }
}

void drawSkeletons(float imgW, float imgH) {
  for (Person person : detectedPeople) {
    noFill();
    strokeWeight(3);
    
    // Draw bbox
    stroke(255, 255, 0);
    float ymin = person.bbox[0] * imgH;
    float xmin = person.bbox[1] * imgW;
    float ymax = person.bbox[2] * imgH;
    float xmax = person.bbox[3] * imgW;
    rectMode(CORNERS);
    rect(xmin, ymin, xmax, ymax);
    
    // Draw keypoints
    stroke(0, 255, 0);
    fill(0, 255, 0);
    for (Keypoint kp : person.keypoints) {
      if (kp.score > 0.2) {
        circle(kp.x * imgW, kp.y * imgH, 10);
      }
    }
    
    // Draw connections
    stroke(255, 0, 0);
    for (int[] conn : CONNECTIONS) {
      Keypoint k1 = person.keypoints.get(conn[0]);
      Keypoint k2 = person.keypoints.get(conn[1]);
      if (k1.score > 0.2 && k2.score > 0.2) {
        line(k1.x * imgW, k1.y * imgH, k2.x * imgW, k2.y * imgH);
      }
    }
  }
}


// Data classes
class Person {
  float confidence;
  float[] bbox = new float[4];
  ArrayList<Keypoint> keypoints = new ArrayList<Keypoint>();
}

class Keypoint {
  float x, y, score;
  Keypoint(float x, float y, float score) {
    this.x = x;
    this.y = y;
    this.score = score;
  }
}
