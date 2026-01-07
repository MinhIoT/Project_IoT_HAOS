import paho.mqtt.client as mqtt
import json
import threading
import time
import traceback


class MQTTClientHAOS:
    """
    MQTT Client an toàn cho AI Camera Gateway

    - MQTT server OFF -> Web + AI vẫn chạy
    - Auto reconnect (paho internal)
    - Publish / Subscribe best-effort
    - Không raise exception ra ngoài
    """

    def __init__(
        self,
        broker,
        port=1883,
        username=None,
        password=None,
        client_id="ai_camera_gateway"
    ):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id

        self.client = mqtt.Client(
            client_id=self.client_id,
            clean_session=True
        )

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        # ===== MQTT callbacks =====
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # ===== Auto reconnect (paho native) =====
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

        self._subscriptions = {}   # topic -> callback
        self._running = False
        self.connected = False

    # =====================================================
    # CALLBACKS
    # =====================================================
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print("[MQTT] Connected")

            # Re-subscribe all registered topics
            for topic in self._subscriptions:
                try:
                    client.subscribe(topic)
                except Exception as e:
                    print(f"[MQTT] Subscribe failed ({topic}):", e)
        else:
            print("[MQTT] Connect failed rc =", rc)

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("[MQTT] Disconnected rc =", rc)

    def _on_message(self, client, userdata, msg):
        callback = self._subscriptions.get(msg.topic)
        if not callback:
            return

        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            payload = msg.payload.decode(errors="ignore")

        try:
            callback(payload)
        except Exception:
            print("[MQTT] Callback error:")
            traceback.print_exc()

    # =====================================================
    # INTERNAL LOOP
    # =====================================================
    def _loop(self):
        """
        MQTT network loop
        - Không throw exception
        - Paho tự reconnect
        """
        while self._running:
            try:
                self.client.loop(timeout=1.0)
            except Exception as e:
                print("[MQTT] Loop error:", e)
            time.sleep(0.1)

    # =====================================================
    # PUBLIC API
    # =====================================================
    def connect(self):
        """
        Kết nối MQTT
        - Không raise exception
        - Broker OFF -> chỉ log
        """
        if self._running:
            return

        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self._running = True

            threading.Thread(
                target=self._loop,
                daemon=True
            ).start()

        except Exception as e:
            self.connected = False
            print("[MQTT] Initial connect failed:", e)

    def disconnect(self):
        self._running = False
        try:
            self.client.disconnect()
        except Exception:
            pass

    def subscribe(self, topic, callback):
        """
        Subscribe an toàn
        """
        self._subscriptions[topic] = callback

        if self.connected:
            try:
                self.client.subscribe(topic)
            except Exception as e:
                print(f"[MQTT] Subscribe failed ({topic}):", e)

    def publish(self, topic, payload, retain=False):
        """
        Publish an toàn:
        - MQTT OFF -> bỏ qua
        - Không crash hệ thống
        """
        if not self.connected:
            return

        try:
            if isinstance(payload, dict):
                payload = json.dumps(payload)

            self.client.publish(topic, payload, retain=retain)

        except Exception as e:
            print("[MQTT] Publish failed:", e)
