#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include <WiFi.h>
#include <WebServer.h>

// Wi-Fi credentials
const char* ssid = "...";
const char* password = "...";

// Pin Definitions
#define TFT_CS 5    // Chip select pin
#define TFT_DC 2    // Data/Command pin
#define TFT_RST -1  // Reset pin (not used in this example, leave -1 if unused)
#define BLUE 0x7aadff //Initialize blue color

const int redPin = 25;
const int greenPin = 33;
const int bluePin = 32;

// Initialize Adafruit ILI9341
Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);

// Create an instance of the WebServer class
WebServer server(80);

void setup() {
  // Initialize Serial Monitor
  Serial.begin(9600);

  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);

  // Connect to Wi-Fi
  Serial.println("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting...");
  }
  Serial.println("Connected to Wi-Fi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Initialize display
  tft.begin();
  tft.setRotation(3);  // Rotate display if necessary

  // Start with happy face
  drawSmiley(0);

  // Set up HTTP route to handle incoming requests
  server.on("/", HTTP_GET, handleRoot);
  server.begin();
}

void loop() {
  server.handleClient(); // Handle incoming HTTP requests
}

void handleRoot() {
  // Check if 'happiness' parameter is present in the URL
  if (server.hasArg("happiness")) {
    int happiness = server.arg("happiness").toInt();
    drawSmiley(happiness);
    server.send(200, "text/plain", "Smiley updated!");
  } else {
    server.send(400, "text/plain", "Missing happiness parameter");
  }
}

void drawSmiley(int happiness) {
  // Define a variable to store the current state
  static int current_state = -1;  // Initialize to an invalid state

  // Check if happiness has changed
  if (happiness != current_state) {
    // Clear the screen
    tft.fillScreen(ILI9341_BLACK);

    // Draw the smiley based on happiness level
    if (happiness == 0) {
      tft.fillScreen(ILI9341_RED);

      digitalWrite(redPin, HIGH);
      digitalWrite(greenPin, LOW);
      digitalWrite(bluePin, LOW);

      for (int i = 0; i < 10; i++) {
        tft.drawLine(72, 56 - i, 112, 71 - i, ILI9341_BLACK);
        tft.drawLine(248, 56 - i, 208, 71 - i, ILI9341_BLACK);
      }
      for (int i = 0; i < 10; i++){
        tft.drawLine(140, 135 - i, 180, 135 - i, ILI9341_BLACK);
        tft.drawLine(110, 165 - i, 140, 135 - i, ILI9341_BLACK);
        tft.drawLine(180, 135 - i, 210, 165 - i, ILI9341_BLACK);
      }
    } else if (happiness == 1) {
      tft.fillScreen(ILI9341_GREEN);

      digitalWrite(redPin, LOW);
      digitalWrite(greenPin, HIGH);
      digitalWrite(bluePin, LOW);

      for (int i = 0; i < 10; i++) {
        tft.drawLine(140, 160 - i, 180, 160 - i, ILI9341_BLACK);
        tft.drawLine(110, 130 - i, 140, 160 - i, ILI9341_BLACK);
        tft.drawLine(180, 160 - i, 210, 130 - i, ILI9341_BLACK);
      }
    } else if (happiness == 2) {
      tft.fillScreen(ILI9341_YELLOW);

      digitalWrite(redPin, HIGH);
      digitalWrite(greenPin, HIGH);
      digitalWrite(bluePin, LOW);

      for (int i = 0; i < 10; i++) {
        tft.drawLine(110, 160 - i, 210, 160 - i, ILI9341_BLACK);
      }
    } else if (happiness == 3) {
      tft.fillScreen(BLUE);

      digitalWrite(redPin, LOW);
      digitalWrite(greenPin, LOW);
      digitalWrite(bluePin, HIGH);

      for (int i = 0; i < 10; i++) {
        tft.drawLine(140, 135 - i, 180, 135 - i, ILI9341_BLACK);
        tft.drawLine(110, 165 - i, 140, 135 - i, ILI9341_BLACK);
        tft.drawLine(180, 135 - i, 210, 165 - i, ILI9341_BLACK);
      }
    }
    //Draw eyes
    tft.fillCircle(88, 70, 12, ILI9341_BLACK);
    tft.fillCircle(232, 70, 12, ILI9341_BLACK);

    // Update the current state
    current_state = happiness;
  }
}