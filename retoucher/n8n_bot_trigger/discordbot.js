const { Client, GatewayIntentBits, Events } = require("discord.js");
require("dotenv").config();

// Create a new client instance
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

// N8N webhook URL - Replace with your actual webhook URL
const N8N_WEBHOOK_URL = process.env.N8N_WEBHOOK_URL;

// When the client is ready, run this code (only once)
client.once(Events.ClientReady, (readyClient) => {
  console.log(`Ready! Logged in as ${readyClient.user.tag}`);
});

// Listen for new messages
client.on(Events.MessageCreate, async (message) => {
  // Ignore messages from bots
  if (message.author.bot) return;

  try {
    // Send message data to n8n webhook
    const response = await fetch(N8N_WEBHOOK_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messageId: message.id,
        channelId: message.channel.id,
        content: message.content,
        attachments: message.attachments.map((attachment) => ({
          url: attachment.url,
          filename: attachment.name,
        })),
        author: {
          id: message.author.id,
          username: message.author.username,
        },
        timestamp: message.createdTimestamp,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  } catch (error) {
    console.error("Error sending webhook:", error);
  }
});

// Listen for message updates
client.on(Events.MessageUpdate, async (oldMessage, newMessage) => {
  // Ignore messages from bots
  if (newMessage.author.bot) return;

  try {
    // Send updated message data to n8n webhook
    const response = await fetch(N8N_WEBHOOK_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messageId: newMessage.id,
        channelId: newMessage.channel.id,
        content: newMessage.content,
        attachments: newMessage.attachments.map((attachment) => ({
          url: attachment.url,
          filename: attachment.name,
        })),
        author: {
          id: newMessage.author.id,
          username: newMessage.author.username,
        },
        timestamp: newMessage.editedTimestamp,
        isUpdate: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  } catch (error) {
    console.error("Error sending webhook:", error);
  }
});

// Login to Discord with your client's token
client.login(process.env.DISCORD_TOKEN);
