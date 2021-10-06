const int ledPin = 2;

void setup() {
  // put your setup code here, to run once:
  pinMode(ledPin, OUTPUT);
  Serial.begin(115200);
  Serial.println("Startup");
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println("Hello there");
  digitalWrite(ledPin, HIGH);
  delay(500);
}
