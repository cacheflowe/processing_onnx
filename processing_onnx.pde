
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
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

// ONNX input info
String inputName;
long[] inputShape;
int inputW = 256;
int inputH = 256;

// Threading for async inference
ExecutorService inferenceExecutor;
AtomicBoolean inferenceRunning = new AtomicBoolean(false);
volatile ArrayList<Person> latestResults = null;
volatile boolean newResultsAvailable = false;

// Reusable buffer for inference thread (only accessed by single inference thread)
IntBuffer inferenceBuffer = null;
int inferenceCount = 0;
long lastInferenceTime = 0;

// Webcam
PImage testImg;
Capture webcam;
volatile PImage latestWebcamFrame = null;  // Latest frame from async callback
volatile boolean newWebcamFrame = false;

// UI controls
float minConfidence = 0.3f;

// Skeleton connections (MoveNet keypoint pairs)
int[][] CONNECTIONS = {
  {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 7}, {7, 9},
  {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

void settings() {
  size(1280, 720);
  pixelDensity(1); // leaving default on Mac massively degrades performance
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
    webcam = new Capture(this, cameras[0]);
    webcam.start();
  }
}

PImage getInputImage() {
  // Return latest webcam frame (non-blocking)
  if (webcam != null && latestWebcamFrame != null) {
    return latestWebcamFrame;
  }
  newWebcamFrame = true; // always true for static image
  return testImg;
}

// Async callback - called by Processing when new webcam frame is ready
void captureEvent(Capture video) {
  video.read();
  latestWebcamFrame = video.copy();  // Copy so we don't hold reference to video buffer
  newWebcamFrame = true;
}

void initializeONNX() {
  // Init ONNX session
  try {
    println("Available ONNX providers: " + OrtEnvironment.getAvailableProviders());
    env = OrtEnvironment.getEnvironment();
    SessionOptions sessionOptions = new SessionOptions();
    
    // Load model from data folder
    String modelPath = dataPath("movenet-multipose-lightning.onnx");
    println("Loading model from: " + modelPath);
    session = env.createSession(modelPath, sessionOptions);
    inputName = session.getInputNames().iterator().next();
    
    // log input/output info
    println("Input info: " + session.getInputInfo());
    println("Output info: " + session.getOutputInfo());
  } catch (OrtException e) {
    e.printStackTrace();
  }
  
  // Pre-allocate input shape
  inputShape = new long[]{1, inputW, inputH, 3};
  inputResized = createImage(inputW, inputH, ARGB);
  
  // Create single-thread executor for inference
  inferenceExecutor = Executors.newSingleThreadExecutor();
}

void draw() {
  background(0);  
  PImage inputImage = getInputImage();
  image(inputImage, 0, 0);
  runMovenet(inputImage);
  drawSkeletons(inputImage.width, inputImage.height);
  drawUI();
}

void runMovenet(PImage inputImage) {
  if (session != null && inputImage != null && inputImage.width > 10) {
    
    // Only submit new inference if previous one is done AND we have a new frame
    if (!inferenceRunning.get() && newWebcamFrame) {
      newWebcamFrame = false;
      // Copy source image to input buffer, resizing to input size
      inputResized.copy(inputImage, 0, 0, inputImage.width, inputImage.height, 0, 0, inputW, inputH);
      submitInference(inputResized);
    }
    
    // Check for new results using flag
    if (newResultsAvailable) {
      detectedPeople = latestResults;
      newResultsAvailable = false;
    } 
  }
}

void drawUI() {
  // background
  rectMode(CORNER);
  noStroke();
  fill(0, 150);
  rect(0, height - 100, 350, 100);
  // text
  fill(255);
  textSize(12);
  text("Min Confidence: " + nf(minConfidence, 1, 2) + " (UP/DOWN arrows)", 20, height - 75);
  text("Render FPS: " + nf(frameRate, 2, 1), 20, height - 35);
  float inferenceRate = (lastInferenceTime > 0) ? 1000.0 / lastInferenceTime : 0;
  text("Inference: " + lastInferenceTime + "ms (" + nf(inferenceRate, 2, 1) + " fps) | Count: " + inferenceCount, 20, height - 15);
}

void keyPressed() {
  if (keyCode == UP) {
    minConfidence = constrain(minConfidence + 0.05, 0, 1);
  }
  if (keyCode == DOWN) {
    minConfidence = constrain(minConfidence - 0.05, 0, 1);
  }
}

void submitInference(PImage img) {
  img.loadPixels();
  final int[] imgPixels = img.pixels;
  final int imgWidth = img.width;
  final int imgHeight = img.height;
  final float capturedMinConf = minConfidence;
  
  inferenceRunning.set(true);
  inferenceExecutor.submit(new Runnable() {
    public void run() {
      try {
        runMovenetAsync(imgPixels, imgWidth, imgHeight, capturedMinConf);
      } finally {
        inferenceRunning.set(false);
      }
    }
  });
}

void runMovenetAsync(int[] pixels, int imgWidth, int imgHeight, float minConf) {
  long startTime = millis();
  OnnxTensor tensor = null;
  Result result = null;
  try {
    // Init buffer once (safe because single-thread executor)
    int bufferSize = imgWidth * imgHeight * 3;
    if (inferenceBuffer == null || inferenceBuffer.capacity() < bufferSize) {
      inferenceBuffer = ByteBuffer.allocateDirect(bufferSize * 4)
        .order(ByteOrder.nativeOrder())
        .asIntBuffer();
    }

    // Prepare input buffer (RGB) with updated pixel data
    inferenceBuffer.clear();  
    for (int i = 0; i < pixels.length; i++) {
      int pixel = pixels[i];
      inferenceBuffer.put((pixel >> 16) & 0xFF); // R
      inferenceBuffer.put((pixel >> 8) & 0xFF);  // G
      inferenceBuffer.put(pixel & 0xFF);         // B
    }
    inferenceBuffer.rewind();
    
    // Create input tensor and run inference
    tensor = OnnxTensor.createTensor(env, inferenceBuffer, inputShape);
    result = session.run(Collections.singletonMap(inputName, tensor));
    
    // Parse output tensor
    float[][][] output = (float[][][]) result.get(0).getValue();
    float[][] people = output[0];
    
    // Parse skeleton results into Java objects
    ArrayList<Person> newResults = new ArrayList<Person>();
    for (int i = 0; i < people.length; i++) {
      float[] personData = people[i];
      float confidence = personData[55];
      if (confidence >= minConf) {
        // Create new Person object w/confidence reading
        Person person = new Person();
        person.confidence = confidence;

        // pull bbox out of ONNX output
        person.bbox[0] = personData[51];
        person.bbox[1] = personData[52];
        person.bbox[2] = personData[53];
        person.bbox[3] = personData[54];
        
        // pull keypoints out of ONNX output
        for (int k = 0; k < 17; k++) {
          float y = personData[k * 3];
          float x = personData[k * 3 + 1];
          float score = personData[k * 3 + 2];
          person.keypoints.add(new Keypoint(x, y, score));
        }
        newResults.add(person);
      }
    }
    
    // Atomic swap with flag
    latestResults = newResults;
    newResultsAvailable = true;
    
    // Track timing
    lastInferenceTime = millis() - startTime;
    inferenceCount++;
    
  } catch (OrtException e) {
    e.printStackTrace();
  } finally {
    // Always close resources to prevent memory leak
    if (result != null) {
      try { result.close(); } catch (Exception e) {}
    }
    if (tensor != null) {
      try { tensor.close(); } catch (Exception e) {}
    }
  }
}

void drawSkeletons(float imgW, float imgH) {
  for (Person person : detectedPeople) {
    noFill();
    strokeWeight(3);
    
    // Draw bbox
    stroke(255, 255, 0);
    rectMode(CORNER);
    float ymin = person.bbox[0] * imgH;
    float xmin = person.bbox[1] * imgW;
    float ymax = person.bbox[2] * imgH;
    float xmax = person.bbox[3] * imgW;
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

void dispose() {
  // Clean up resources on exit
  // First, stop accepting new tasks and wait for current inference to complete
  if (inferenceExecutor != null) {
    inferenceExecutor.shutdown(); // Stop accepting new tasks
    try {
      // Wait for current inference to finish (up to 5 seconds)
      if (!inferenceExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
        inferenceExecutor.shutdownNow(); // Force shutdown if taking too long
        inferenceExecutor.awaitTermination(2, TimeUnit.SECONDS);
      }
    } catch (InterruptedException e) {
      inferenceExecutor.shutdownNow();
      Thread.currentThread().interrupt();
    }
  }
  
  // Now it's safe to close ONNX resources (no inference running)
  if (session != null) {
    try { session.close(); } catch (Exception e) { e.printStackTrace(); }
  }
  if (env != null) {
    try { env.close(); } catch (Exception e) { e.printStackTrace(); }
  }
}
