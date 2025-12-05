import websocket
import threading
import time
import json
import logging
import uuid
import queue
import traceback
from collections import deque
import websockets
import asyncio
from config import is_print_debug
from utilities.yxRegistry import YxRegistry
import copy

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def all_not_none(*args):
    return all(arg is not None for arg in args)

#websocket服务器
class YXWebSocketServer:
    def __init__(self, host='localhost', port=9003):
        self.host = host
        self.port = port
        self.server = None
        self.loop = None
        self.thread = None
        self.running = False
        self.funTool = None

    async def handler(self, websocket):
        """处理 WebSocket 连接"""
        client_ip = websocket.remote_address[0]
        logger.info(f"客户端连接: {client_ip}")

        try:
            async for message in websocket:
                if is_print_debug:
                    logger.info(f"收到来自 {client_ip} 的消息: {message}")
                if self.funTool is not None:
                    json_dict = json.loads(message)
                    if isinstance(json_dict, dict):
                        if 'fun' in json_dict:
                            fun = json_dict['fun']
                            if fun == 'doTask':
                                if 'name' in json_dict:
                                    func_name = json_dict['name']
                                    try:
                                        raw_observation = copy.deepcopy(self.funTool.get_sim_data())
                                        YxRegistry.call(json_dict['name'], json_dict['targets'], raw_observation)
                                    except AttributeError:
                                        print(f"指令不存在：{func_name}")
                                    except Exception as e:
                                        print(f"指令执行失败：{func_name}，错误：{str(e)}")
                                else:
                                    print("收到错误的doTask格式")
                            elif fun == 'heart':
                                func_data = {
                                    "fun": "taskList",
                                    "taskList": YxRegistry.get_functions()
                                }
                                json_data = json.dumps(func_data)
                                await websocket.send(json_data)
                            elif fun == 'cancelTask':
                                YxRegistry.call(fun, json_dict['target'], None)

        except websockets.exceptions.ConnectionClosed:
            if is_print_debug:
                logger.info(f"客户端断开连接: {client_ip}")

    def start_server(self):
        """启动服务器"""

        async def server_main():
            self.server = await websockets.serve(self.handler, self.host, self.port)
            logger.info(f"WebSocket 服务器启动在 ws://{self.host}:{self.port}")
            await self.server.wait_closed()

        self.running = True
        asyncio.run(server_main())

    def start(self):
        """在新线程中启动服务器"""
        self.thread = threading.Thread(target=self.start_server, daemon=True)
        self.thread.start()
        logger.info("WebSocket 服务器线程已启动")

    def stop(self):
        """停止服务器"""
        if self.server:
            self.server.close()
        self.running = False
        logger.info("WebSocket 服务器已停止")


#websocket客户端
class YXWebSocketClient:
    def __init__(self, url,
                 on_message=None,
                 on_open=None,
                 on_close=None,
                 on_error=None,
                 reconnect_interval=3,
                 heartbeat_interval=25,
                 max_retry=5):
        """
        可靠的WebSocket客户端，确保连接稳定性

        :param url: WebSocket服务器URL
        :param on_message: 消息处理回调
        :param on_open: 连接打开回调
        :param on_close: 连接关闭回调
        :param on_error: 错误处理回调
        :param reconnect_interval: 重连间隔(秒)
        :param heartbeat_interval: 心跳间隔(秒)
        :param max_retry: 最大重连次数
        """
        self.url = url
        self.client_id = f"ws-{uuid.uuid4().hex[:8]}"
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        self.max_retry = max_retry
        self.retry_count = 0

        # 回调函数
        self.on_message = on_message or self._default_on_message
        self.on_open = on_open or self._default_on_open
        self.on_close = on_close or self._default_on_close
        self.on_error = on_error or self._default_on_error

        # 连接状态
        self.is_connected = False
        self.is_active = True
        self.should_reconnect = True

        # 消息队列
        self.send_queue = queue.Queue(maxsize=1000)
        self.received_messages = deque(maxlen=100)

        # 连接线程
        self.thread = None
        self._start_connection_thread()
        if is_print_debug:
            logger.info(f"客户端 {self.client_id} 初始化完成，准备连接 {url}")

    def _start_connection_thread(self):
        """启动连接线程"""
        self.thread = threading.Thread(target=self._connection_manager, daemon=True)
        self.thread.name = f"WS-Conn-{self.client_id}"
        self.thread.start()

    def _connection_manager(self):
        """管理连接和重连逻辑"""
        while self.is_active:
            try:
                # 创建WebSocket实例
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open_wrapper,
                    on_message=self._on_message_wrapper,
                    on_error=self._on_error_wrapper,
                    on_close=self._on_close_wrapper
                )
                if is_print_debug:
                    logger.info(f"[{self.client_id}] 正在连接 {self.url}...")

                # 运行WebSocket，禁用内置SSL验证
                self.ws.run_forever(
                    skip_utf8_validation=True,
                    ping_interval=self.heartbeat_interval,
                    ping_timeout=20
                )

                # 连接断开后的处理
                if self.should_reconnect and self.is_active:
                    if self.retry_count < self.max_retry or self.max_retry == 0:
                        self.retry_count += 1
                        logger.warning(
                            f"[{self.client_id}] 连接中断，{self.reconnect_interval}秒后尝试重连 ({self.retry_count}/{self.max_retry})")
                        time.sleep(self.reconnect_interval)
                    else:
                        logger.error(f"[{self.client_id}] 达到最大重连次数 {self.max_retry}，停止尝试")
                        self.is_active = False
                        break

            except Exception as e:
                logger.error(f"[{self.client_id}] 连接管理器异常: {e}")
                if self.is_active:
                    time.sleep(self.reconnect_interval)

    # 回调函数包装器
    def _on_open_wrapper(self, ws):
        self.is_connected = True
        self.retry_count = 0  # 重置重连计数器
        if is_print_debug:
            logger.info(f"[{self.client_id}] 成功连接到 {self.url}")

        # 启动发送线程
        threading.Thread(target=self._message_sender, daemon=True).start()

        # 调用用户自定义回调
        try:
            self.on_open(ws)
        except Exception as e:
            logger.error(f"[{self.client_id}] on_open回调异常: {e}")

    def _on_message_wrapper(self, ws, message):
        try:
            # 添加到最近消息队列
            self.received_messages.append(message)

            # 调用用户回调
            self.on_message(ws, message)
        except Exception as e:
            data = json.loads(message)
            traceback.print_exc()
            logger.error(f"[{self.client_id}] on_message回调异常: {e}")

    def _on_error_wrapper(self, ws, error):
        if not self.is_connected:
            # 连接错误处理
            logger.warning(f"[{self.client_id}] 连接错误: {error}")
        else:
            logger.error(f"[{self.client_id}] WebSocket错误: {error}")
        try:
            self.on_error(ws, error)
        except Exception as e:
            logger.error(f"[{self.client_id}] on_error回调异常: {e}")

    def _on_close_wrapper(self, ws, close_status_code, close_msg):
        self.is_connected = False
        if close_status_code or close_msg:
            logger.info(f"[{self.client_id}] 连接关闭: 状态码={close_status_code}, 消息={close_msg}")
        else:
            logger.info(f"[{self.client_id}] 连接关闭")

        try:
            self.on_close(ws, close_status_code, close_msg)
        except Exception as e:
            logger.error(f"[{self.client_id}] on_close回调异常: {e}")

    def _message_sender(self):
        """在专用线程中发送消息"""
        while self.is_connected:
            try:
                # 从队列获取消息，带有超时以避免阻塞
                message = self.send_queue.get(timeout=0.5)
                if self.ws and self.is_connected:
                    try:
                        if isinstance(message, dict):
                            # 自动转换字典为JSON
                            self.ws.send(json.dumps(message))
                        else:
                            self.ws.send(message)
                        if is_print_debug:
                            logger.debug(f"[{self.client_id}] 消息发送成功")
                    except websocket.WebSocketConnectionClosedException:
                        logger.warning(f"[{self.client_id}] 发送失败: 连接已关闭")
                        self.send_queue.put(message)  # 将消息放回队列以便稍后发送
                        time.sleep(0.5)
                        break
                    except Exception as e:
                        logger.error(f"[{self.client_id}] 发送异常: {e}")
                else:
                    # 连接未就绪，将消息放回队列
                    self.send_queue.put(message)
                    time.sleep(0.5)

            except queue.Empty:
                # 队列为空，继续检查
                continue
            except Exception as e:
                logger.error(f"[{self.client_id}] 发送线程异常: {e}")
                if self.is_connected:
                    time.sleep(1)

    # 默认回调函数
    def _default_on_message(self, ws, message):
        """默认消息处理器"""
        logger.info(f"[{self.client_id}] 收到消息: {message[:80]}{'...' if len(message) > 80 else ''}")

    def _default_on_open(self, ws):
        logger.info(f"[{self.client_id}] 连接已打开")

    def _default_on_error(self, ws, error):
        logger.error(f"[{self.client_id}] 发生错误: {error}")

    def _default_on_close(self, ws, status, msg):
        logger.info(f"[{self.client_id}] 连接关闭: {status} - {msg}")

    def send(self, message):
        """发送消息到服务器"""
        if not self.is_active:
            logger.warning(f"[{self.client_id}] 尝试发送消息但客户端已停止")
            return False

        try:
            # 直接处理基本类型
            if isinstance(message, (str, bytes, dict)):
                self.send_queue.put(message)
                return True
            elif hasattr(message, "to_dict") and callable(message.to_dict):
                # 处理自定义对象 (如果有to_dict方法)
                self.send_queue.put(message.to_dict())
                return True
            else:
                # 尝试序列化其他类型
                self.send_queue.put(str(message))
                return True
        except queue.Full:
            logger.warning(f"[{self.client_id}] 发送队列已满，丢弃消息")
            return False
        except Exception as e:
            logger.error(f"[{self.client_id}] 发送失败: {e}")
            return False

    def get_last_messages(self, count=5):
        """获取最近收到的消息"""
        return list(self.received_messages)[-count:]

    def close(self):
        """安全关闭连接"""
        self.is_active = False
        self.should_reconnect = False

        if hasattr(self, 'ws') and self.ws:
            try:
                self.ws.close()
            except:
                pass

        logger.info(f"[{self.client_id}] 客户端已关闭")


# WebSocket管理器 - 简化多实例管理
class YXWebSocketClientManager:
    def __init__(self):
        self.clients = {}
        self.lock = threading.Lock()

    def add_client(self, url, **kwargs):
        """添加新的WebSocket客户端"""
        with self.lock:
            if url in self.clients:
                logger.warning(f"URL {url} 已有客户端")
                return self.clients[url]

            client = YXWebSocketClient(url, **kwargs)
            self.clients[url] = client
            return client

    def remove_client(self, url):
        """添加新的WebSocket客户端"""
        with self.lock:
            if url in self.clients:
                self.clients[url].close()
                self.clients.pop(url)

    def send(self, url, message):
        """通过指定URL的客户端发送消息"""
        with self.lock:
            client = self.clients.get(url)
            if client and client.is_active:
                return client.send(message)
            return False

    def get_client(self, url):
        """获取指定URL的客户端"""
        with self.lock:
            return self.clients.get(url)

    def close_all(self):
        """关闭所有客户端"""
        with self.lock:
            for url, client in self.clients.items():
                client.close()
            self.clients.clear()
        logger.info("所有WebSocket客户端已关闭")

    def stop_all(self):
        """停止所有客户端"""
        with self.lock:
            for url, client in list(self.clients.items()):
                client.close()
                self.clients.pop(url, None)
        logger.info("所有WebSocket客户端已停止")