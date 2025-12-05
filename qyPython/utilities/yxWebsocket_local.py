import threading
import json
import logging
import websockets
import asyncio
from config import is_print_debug


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def all_not_none(*args):
    return all(arg is not None for arg in args)
#websocket服务器
class YXWebSocketServerMain:
    def __init__(self, host='localhost', port=9001):
        self.host = host
        self.port = port
        self.server = None
        self.loop = None
        self.thread = None
        self.running = False
        self.funTool = None
        self.wait_dict = {}
        self.sim_data_dict = {}

        self.websocket_clients = {}
        self.lock = threading.Lock()

        self.process_num = 0
        self.dict_lock = threading.Lock()

        self.async_loop = asyncio.new_event_loop()
        t = threading.Thread(target=self.start_async_loop, args=(self.async_loop,), daemon=True)
        t.start()

    async def async_send_to_room(self, room_id, message):

        try:
            await self.send_to_room(room_id, message)  # 模拟异步IO操作
        except Exception as e:
            print('主进程：与实例断开连接')

    def start_async_loop(self,loop):
        """在后台线程中启动事件循环"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def send_message(self, room_id, message):
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.async_send_to_room(room_id, message),
                self.async_loop
            )
        except KeyboardInterrupt:
            print("正在退出...")

    def start_server(self):
        """启动服务器"""
        async def handler(websocket):
            """处理 WebSocket 连接"""
            try:
                async for message in websocket:
                    self.handle_message(message, websocket)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"客户端断开连接")

        async def server_main():
            self.server = await websockets.serve(handler, self.host, self.port)
            if is_print_debug:
                logger.info(f"服务器已启动")
            self.running = True
            await self.server.wait_closed()
        asyncio.run(server_main())

    def start(self):
        """在新线程中启动服务器"""
        self.thread = threading.Thread(target=self.start_server, daemon=True)
        self.thread.start()
        if is_print_debug:
            logger.info("WebSocket 服务器线程已启动")

    def stop(self):
        """停止服务器"""
        if self.server:
            self.server.close()
        self.running = False
        if is_print_debug:
            logger.info("WebSocket 服务器已停止")

    def handle_message(self, message, ws):
        try:
            if message != 'a':
                tjson = json.loads(message)
                if isinstance(tjson, dict):
                    if 'fun' in tjson:
                        fun = tjson['fun']
                        room_id = tjson['roomId']
                        if is_print_debug:
                            logger.info(f"收到实例{room_id}的消息: {message}")
                        if room_id not in self.websocket_clients:
                            self.websocket_clients[room_id] = ws
                        if fun == 'waitNextStep':
                            self.wait_dict[room_id] = True
                            sim_data = tjson['simData']
                            with self.dict_lock:
                                self.sim_data_dict[room_id] = sim_data
                        elif fun == 'simulationCompleted':
                            self.wait_dict.pop(room_id)
                            self.process_num -= 1

        except json.JSONDecodeError:
            logger.warning(f"收到的消息不是有效的JSON: {message}")

    async def send_to_room(self, room_id, message):
        if room_id in self.websocket_clients:
            client = self.websocket_clients[room_id]
            try:
                json_data = json.dumps(message, ensure_ascii=False, separators=(',', ':'))
                await client.send(json_data)
                if is_print_debug:
                    print(f"消息已发送至实例 {room_id}")
            except websockets.ConnectionClosed:
                print(f"尝试发送时发现实例 {room_id} 的连接已断开")
                async with self.lock:
                    if room_id in self.websocket_clients:
                        del self.websocket_clients[room_id]
                        logger.info(f"已移除房间 {room_id} 的客户端")
                # 清理相关状态
                with self.dict_lock:
                    self.wait_dict.pop(room_id, None)
                    self.sim_data_dict.pop(room_id, None)
                # del self.websocket_clients[room_id]
        else:
            print(f"实例 {room_id} 未找到活跃连接")