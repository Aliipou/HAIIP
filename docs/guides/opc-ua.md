# OPC UA Integration Guide

## Connecting to a PLC

```python
from asyncua import Client

async def connect_plc(endpoint: str) -> Client:
    client = Client(url=endpoint)
    await client.connect()
    return client

# Example: Siemens S7-1500 via OPC UA
client = await connect_plc("opc.tcp://192.168.1.100:4840")
```

## Reading Node Values

```python
async def read_sensor_data(client: Client, node_ids: list[str]) -> dict[str, float]:
    results = {}
    for node_id in node_ids:
        node = client.get_node(node_id)
        value = await node.read_value()
        results[node_id] = float(value)
    return results

# Example node IDs (Siemens convention)
node_ids = [
    "ns=3;s=Motor1.Speed",
    "ns=3;s=Motor1.Temperature",
    "ns=3;s=Pump1.Pressure",
]
```

## Subscription (Push Mode)

```python
from asyncua import ua

class SensorHandler:
    def datachange_notification(self, node, val, data):
        print(f"Node {node}: {val}")

handler = SensorHandler()
subscription = await client.create_subscription(500, handler)  # 500ms interval
await subscription.subscribe_data_change(nodes)
```

## HAIIP Configuration

```yaml
# config/opcua.yml
endpoints:
  - name: press-line-1
    url: opc.tcp://press-plc-01:4840
    poll_interval_ms: 1000
    nodes:
      - id: "ns=3;s=Press.Force"
        tag: press_force
      - id: "ns=3;s=Press.Temperature"
        tag: press_temp
```
