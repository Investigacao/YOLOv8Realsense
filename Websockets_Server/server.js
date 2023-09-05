const WebSocket = require("ws");

// Create a WebSocket server
const wss = new WebSocket.Server({ port: 3306 });

// Define a map to store the clients and their subscribed topics
const clients = new Map();

// Handle a new connection
wss.on("connection", function (ws) {
  console.log("A new client connected");

  // Initialize an array to store subscribed topics for this client
  const subscribedTopics = [];

  // Function to broadcast a message to all topics a client is subscribed to
  function broadcastToSubscribedTopics(message) {
    subscribedTopics.forEach((topic) => {
      if (clients.has(topic)) {
        clients.get(topic).forEach((client) => {
          if (client !== ws) {
            console.log(
              `Sending message to client in topic ${topic}: ${message}`
            );
            client.send(JSON.stringify(message));
          }
        });
      }
    });
  }

  // Handle a message from the client
  ws.on("message", function (message) {
    console.log("Received message: %s", message);

    // Parse the message as JSON
    let parsedMessage;
    try {
      parsedMessage = JSON.parse(message);
    } catch (error) {
      console.error("Error parsing message:", error);
      return;
    }

    // Check if the message contains a 'topic' field
    if (parsedMessage && parsedMessage.topic) {
      const { topic, data } = parsedMessage;

      // Store the client in the map for this topic
      if (!clients.has(topic)) {
        clients.set(topic, new Set());
      }
      clients.get(topic).add(ws);
      console.log(`Client subscribed to topic ${topic}`);
      subscribedTopics.push(topic);

      // Broadcast the message to the specified topic
      broadcastToSubscribedTopics(data);
    } else {
      // No 'topic' field in the message, broadcast to all subscribed topics
      broadcastToSubscribedTopics(parsedMessage);
    }
  });
});

// const WebSocket = require("ws");

// // Create a WebSocket server
// const wss = new WebSocket.Server({ port: 3306 });

// // Define a map to store the clients by topic
// const clientsByTopic = new Map();

// // Broadcast a message to all clients in the topic except for the sender
// function broadcast(topic, sender, message) {
// 	const clients = clientsByTopic.get(topic);
// 	if (!clients) {
// 		return;
// 	}
// 	clients.forEach(function (client) {
// 		if (client !== sender) {
// 			console.log("Sending message to client: %s", message);
// 			client.send(JSON.stringify(message));
// 		}
// 	});
// }

// // Handle a new connection
// wss.on("connection", function (ws) {
// 	console.log("A new client connected");

// 	// Handle a message from the client
// 	ws.on("message", function (message) {
// 		console.log("Received message: %s", message);

// 		// Get the topic of the message
// 		// const topic = message.split(":")[0];

// 		const { topic, data } = JSON.parse(message);

// 		// Store the client in the map
// 		if (!clientsByTopic.has(topic)) {
// 			clientsByTopic.set(topic, new Set());
// 		}
// 		clientsByTopic.get(topic).add(ws);
// 		console.log(`Clients in topic ${topic}: ${clientsByTopic.get(topic).size}`);

// 		// Broadcast the message to all clients in the topic except for the sender
// 		broadcast(topic, ws, data);
// 	});
// });
