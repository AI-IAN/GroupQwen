"""
WebSocket Handler for Real-time Streaming

Handles real-time streaming of model responses via WebSocket.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import logging
import asyncio

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for streaming responses.
    """

    def __init__(self):
        # Active connections: {client_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}

        # Client subscriptions: {client_id: {subscription_ids}}
        self.subscriptions: Dict[str, Set[str]] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        """
        Accept and register a new WebSocket connection.

        Args:
            client_id: Unique client identifier
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()

        logger.info(f"Client connected: {client_id}")

        await self.send_personal_message(
            client_id,
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id
            }
        )

    def disconnect(self, client_id: str):
        """
        Remove a WebSocket connection.

        Args:
            client_id: Client identifier
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        if client_id in self.subscriptions:
            del self.subscriptions[client_id]

        logger.info(f"Client disconnected: {client_id}")

    async def send_personal_message(self, client_id: str, message: Dict):
        """
        Send message to specific client.

        Args:
            client_id: Target client
            message: Message dict
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: Dict):
        """
        Broadcast message to all connected clients.

        Args:
            message: Message dict
        """
        disconnected = []

        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

    async def stream_response(self, client_id: str, response_id: str):
        """
        Stream a response to a client in chunks.

        Args:
            client_id: Target client
            response_id: Response identifier
        """
        # Placeholder for streaming logic
        # In production, would stream from model inference

        chunks = [
            "This is ",
            "a streaming ",
            "response ",
            "from the ",
            "model."
        ]

        for i, chunk in enumerate(chunks):
            await self.send_personal_message(
                client_id,
                {
                    "type": "stream",
                    "response_id": response_id,
                    "chunk": chunk,
                    "index": i,
                    "done": i == len(chunks) - 1
                }
            )

            await asyncio.sleep(0.1)  # Simulate streaming delay

    def subscribe(self, client_id: str, subscription_id: str):
        """
        Subscribe client to a specific stream.

        Args:
            client_id: Client identifier
            subscription_id: Subscription identifier
        """
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(subscription_id)

    def unsubscribe(self, client_id: str, subscription_id: str):
        """
        Unsubscribe client from a stream.

        Args:
            client_id: Client identifier
            subscription_id: Subscription identifier
        """
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(subscription_id)

    def get_connection_count(self) -> int:
        """
        Get number of active connections.

        Returns:
            Connection count
        """
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket, client_id: str):
    """
    Handle WebSocket connection lifecycle.

    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await manager.connect(client_id, websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "ping":
                    # Respond to ping
                    await manager.send_personal_message(
                        client_id,
                        {"type": "pong"}
                    )

                elif message_type == "subscribe":
                    # Subscribe to stream
                    subscription_id = message.get("subscription_id")
                    manager.subscribe(client_id, subscription_id)

                    await manager.send_personal_message(
                        client_id,
                        {
                            "type": "subscribed",
                            "subscription_id": subscription_id
                        }
                    )

                elif message_type == "unsubscribe":
                    # Unsubscribe from stream
                    subscription_id = message.get("subscription_id")
                    manager.unsubscribe(client_id, subscription_id)

                    await manager.send_personal_message(
                        client_id,
                        {
                            "type": "unsubscribed",
                            "subscription_id": subscription_id
                        }
                    )

                elif message_type == "request_stream":
                    # Request streaming response
                    response_id = message.get("response_id")
                    await manager.stream_response(client_id, response_id)

                else:
                    logger.warning(f"Unknown message type: {message_type}")

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from client {client_id}")

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)
