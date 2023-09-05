const WebSocket = require("ws");

// Define the WebSocket server URL
const serverUrl = "ws://localhost:3306";

// Create a WebSocket client
const ws = new WebSocket(serverUrl);

// Handle the connection event
ws.on("open", function () {
  // Send the initial message to join a topic
  const initialMessage = JSON.stringify({
    topic: "robot",
    data: "Hello, WebSocket Server!",
  });
  ws.send(initialMessage);
  console.log("Connected to WebSocket server.");
});

// Handle incoming messages from the server
ws.on("message", function (message) {
  console.log("Received message from server: " + message);
});

// Handle errors
ws.on("error", function (error) {
  console.error("WebSocket error: " + error);
});
