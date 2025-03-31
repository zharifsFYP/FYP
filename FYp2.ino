#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ===== Define XIAO ESP32S3 Sense Camera Pins =====
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13


#define TRIG_PIN 2
#define ECHO_PIN 3


const float thresholdDistance = 10.0; // cm

const char* ssid = "Your Connection";
const char* password = "your password";

//web server on port 80
WebServer server(80);

// Function to measure distance using the ultrasonic sensor
float measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH, 10000);
  Serial.print("Raw pulse duration: ");
  Serial.println(duration);
  
  if (duration < 10) {
    return 999.0;  //No valid reading
  }
  
  float distance = (duration * 0.0343) / 2;
  return distance;
}

// HTTP handler for /capture
void handleCapture() {
  float distance = measureDistance();
  Serial.print("Measured Distance: ");
  Serial.print(distance);
  Serial.println(" cm");
  
  if (distance > thresholdDistance) {
    String msg = "Ultrasonic sensor not triggered. Current distance: " + String(distance) + " cm";
    server.send(200, "text/plain", msg);
    return;
  }
  
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  
  server.sendHeader("Content-Type", "image/jpeg");
  // Use send_P to send the image buffer directly
  server.send_P(200, "image/jpeg", (const char*) fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handleSurCapture() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  server.sendHeader("Content-Type", "image/jpeg");
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}


void handleRoot() {
  String html = "<html><body><h1>ESP32 Camera</h1>";
  html += "<p>Go to <a href=\"/capture\">/capture</a> to capture an image.</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

//initialize the camera
bool initCamera() {
  camera_config_t config;
  config.ledc_channel   = LEDC_CHANNEL_0;
  config.ledc_timer     = LEDC_TIMER_0;
  config.pin_d0         = Y2_GPIO_NUM;
  config.pin_d1         = Y3_GPIO_NUM;
  config.pin_d2         = Y4_GPIO_NUM;
  config.pin_d3         = Y5_GPIO_NUM;
  config.pin_d4         = Y6_GPIO_NUM;
  config.pin_d5         = Y7_GPIO_NUM;
  config.pin_d6         = Y8_GPIO_NUM;
  config.pin_d7         = Y9_GPIO_NUM;
  config.pin_xclk       = XCLK_GPIO_NUM;
  config.pin_pclk       = PCLK_GPIO_NUM;
  config.pin_vsync      = VSYNC_GPIO_NUM;
  config.pin_href       = HREF_GPIO_NUM;
  config.pin_sscb_sda   = SIOD_GPIO_NUM;
  config.pin_sscb_scl   = SIOC_GPIO_NUM;
  config.pin_pwdn       = PWDN_GPIO_NUM;
  config.pin_reset      = RESET_GPIO_NUM;
  
  config.xclk_freq_hz   = 10000000;
  config.pixel_format   = PIXFORMAT_JPEG;
  config.frame_size     = FRAMESIZE_VGA;
  config.jpeg_quality   = 12;
  config.fb_count       = 1;
  config.fb_location    = CAMERA_FB_IN_PSRAM;
  if (psramFound()){
    config.fb_count = 2;
  }
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }
  Serial.println("Camera init succeeded!");
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starting ESP32 Camera with Ultrasonic Trigger...");
  
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP Address: ");
  Serial.println(WiFi.localIP());
  
  // Initialize the camera
  if (!initCamera()) {
    Serial.println("Camera initialization failed. Halting.");
    while (true) { delay(1000); }
  }
  
  // Set up HTTP endpoints
  server.on("/", handleRoot);
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/surCapture", HTTP_GET, handleSurCapture);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
  yield();
}
