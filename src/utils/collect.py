"""
    Stream data from gdax to file
"""

import asyncio
import sys
import json
import time
import csv
import websockets
import os

URI = 'wss://ws-feed.gdax.com'
SUBSCRIBE_REQUEST = {
    'type': 'subscribe',
    'product_ids': ['BTC-USD'],
    'channels': [
        'matches',
        'heartbeat']
}

COLUMNS = ['time', 'price', 'size']
NUM_MATCHES = 0
FILEPATH = None


async def _connect(path):
    async with websockets.connect(path) as ws:
        print('connecting')
        await ws.send(json.dumps(SUBSCRIBE_REQUEST))
        print('Connected')
        message = await ws.recv()
        print(' {} '.format(message))
        await _listen(ws)


async def _listen(ws):
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=20)
        except asyncio.TimeoutError:
            # No data in 20 seconds, check the connection.
            try:
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=10)
            except asyncio.TimeoutError:
                # No response to ping in 10 seconds, disconnect.
                break
        else:
            handle_message(json.loads(msg))
    return


def match_counter():
    global NUM_MATCHES
    NUM_MATCHES += 1
    if NUM_MATCHES % 10 == 0:
        sys.stdout.write("\r matches entered: {}".format(NUM_MATCHES))
        sys.stdout.flush()


def handle_message(msg):
    if msg['type'] == 'match':
        match_counter()
        with open(FILEPATH, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([msg[key] for key in COLUMNS])


if __name__ == "__main__":
    time = time.strftime("%Y%m%d-%H%M%S")
    FILEPATH = os.path.join(os.getcwd(), 'data/raw/{}.csv'.format(time))
    asyncio.get_event_loop().run_until_complete(_connect(URI))
