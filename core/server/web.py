"""
图谱可视化 Web 服务
提供实时查看图谱可视化的 Web 界面
"""
import sys
import logging
import os
from pathlib import Path
from typing import Optional
from flask import Flask, render_template, jsonify, request

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

from core import Neo4jStorageManager, EmbeddingClient
from core.log import info as _log_info, error as _log_error

# Load HTML template
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Sub-modules for route registration
from core.server.web_graph import register_graph_routes
from core.server.web_versions import register_version_routes


class GraphWebServer:
    """图谱可视化 Web 服务器"""

    def __init__(self, storage_path: str = "./graph/default", port: int = 5000,
                 embedding_model_path: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 embedding_device: str = "cpu",
                 embedding_use_local: bool = True,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        """
        初始化 Web 服务器

        Args:
            storage_path: 存储路径
            port: 服务器端口
            embedding_model_path: 本地embedding模型路径（优先使用）
            embedding_model_name: HuggingFace embedding模型名称
            embedding_device: 计算设备 ("cpu" 或 "cuda")
            embedding_use_local: 是否优先使用本地模型
        """
        self.storage_path = storage_path
        self._base_storage_path = os.path.dirname(storage_path) if os.path.isdir(storage_path) and not os.path.basename(storage_path) else storage_path
        self.port = port
        self._neo4j_uri = neo4j_uri
        self._neo4j_auth = (neo4j_user, neo4j_password)
        self.app = Flask(__name__)

        # 初始化embedding客户端
        self.embedding_client = EmbeddingClient(
            model_path=embedding_model_path,
            model_name=embedding_model_name,
            device=embedding_device,
            use_local=embedding_use_local
        )

        # 初始化存储
        self.storage = Neo4jStorageManager(
            storage_path,
            neo4j_uri=neo4j_uri,
            neo4j_auth=(neo4j_user, neo4j_password),
            embedding_client=self.embedding_client,
        )

        # 缓存当前使用的存储路径（用于路径切换检测）
        self._current_storage_path = storage_path

        # 设置路由
        self._setup_routes()

    def _switch_storage_path(self, new_path: str):
        """
        切换存储路径

        Args:
            new_path: 新的存储路径（必须在 base_storage_path 目录下）
        """
        if new_path != self._current_storage_path:
            # 安全校验：路径必须在存储根目录下
            base = Path(self._base_storage_path).resolve()
            target = (base / new_path).resolve() if not os.path.isabs(new_path) else Path(new_path).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                raise ValueError(f"存储路径必须在 {base} 目录下: {new_path}")
            try:
                # 重新初始化存储
                self.storage = Neo4jStorageManager(
                    str(target),
                    neo4j_uri=self._neo4j_uri,
                    neo4j_auth=self._neo4j_auth,
                    embedding_client=self.embedding_client,
                )
                self._current_storage_path = str(target)
                logger.info("已切换到新的存储路径: %s", target)
            except Exception as e:
                logger.error("切换存储路径失败: %s", e)
                raise

    def _setup_routes(self):
        """设置 Flask 路由"""

        @self.app.route('/api/status')
        def health():
            """健康检查，与 service_api 响应格式一致。/api/status 为常见监控/前端探测路径别名。"""
            try:
                embedding_available = (
                    self.embedding_client is not None
                    and getattr(self.embedding_client, 'is_available', lambda: True)()
                )
                return jsonify({
                    'success': True,
                    'data': {
                        'storage_path': str(self._current_storage_path),
                        'embedding_available': embedding_available,
                    },
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/')
        def index():
            """主页"""
            return render_template('graph.html')

        # Register routes from sub-modules
        register_graph_routes(self)
        register_version_routes(self)

    def run(self, debug: bool = False, host: str = '0.0.0.0'):
        """
        启动 Web 服务器

        Args:
            debug: 是否开启调试模式
            host: 监听地址
        """
        # 获取embedding模型信息
        embedding_info = "未配置"
        if self.embedding_client.model:
            if self.embedding_client.model_path:
                embedding_info = f"本地模型: {self.embedding_client.model_path}"
            elif self.embedding_client.model_name:
                embedding_info = f"HuggingFace: {self.embedding_client.model_name}"
            else:
                embedding_info = "默认模型: all-MiniLM-L6-v2"
        else:
            embedding_info = "未安装sentence-transformers（将使用文本相似度搜索）"

        _log_info("System", f"""
╔════════════════════════════════════════╗
║   时序记忆图谱可视化 Web 服务            ║


🌐 Web服务器已启动
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
访问地址: http://localhost:{self.port}
API地址:  http://localhost:{self.port}/api/graphs/data

📦 Embedding模型: {embedding_info}
📁 存储路径: {self.storage_path}

提示:
  1. 在浏览器中打开 http://localhost:{self.port}
  2. 图谱会自动加载并显示
  3. 点击"刷新图谱"按钮手动更新
  4. 点击节点或边查看详细信息
  5. 使用搜索功能进行语义搜索

按 Ctrl+C 停止服务器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)

        try:
            self.app.run(host=host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            _log_info("System", "服务器已停止")
        except Exception as e:
            _log_error("System", f"服务器错误: {e}")


def main():
    """主函数。支持 --config 与 service_api 共用 service_config.json。"""
    import argparse
    from core.server.config import load_config, resolve_embedding_model

    parser = argparse.ArgumentParser(description='时序记忆图谱可视化 Web 服务')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（与 service_api 共用 service_config.json 时，将使用其中 storage_path 与 embedding）')
    parser.add_argument('--storage', type=str, default='./graph',
                       help='图谱基础目录 (默认: ./graph，未使用 --config 时生效)')
    parser.add_argument('--graph-id', type=str, default='default',
                       help='要可视化的图谱 ID (默认: default，对应 ./graph/default/)')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口 (默认: 5000，与 service_api 不同端口可同时运行)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                       help='开启调试模式')
    parser.add_argument('--embedding-model-path', type=str, default=None,
                       help='本地 embedding 模型路径（未使用 --config 时生效）')
    parser.add_argument('--embedding-model-name', type=str, default=None,
                       help='HuggingFace embedding 模型名称（例如: all-MiniLM-L6-v2）')
    parser.add_argument('--embedding-device', type=str, default='cpu',
                       help='计算设备 (默认: cpu，可为 cuda 或 cuda:0)')
    parser.add_argument('--embedding-use-local', action='store_true', default=True,
                       help='优先使用本地模型（默认: True）')
    parser.add_argument('--embedding-use-hf', action='store_true', default=False,
                       help='优先使用 HuggingFace 模型（与 --embedding-use-local 互斥）')

    args = parser.parse_args()

    if args.config:
        if not Path(args.config).exists():
            _log_error("System", f"配置文件不存在: {args.config}")
            return 1
        config = load_config(args.config)
        base_path = config.get('storage_path', args.storage)
        emb_cfg = config.get('embedding') or {}
        emb_path, emb_name, emb_use_local = resolve_embedding_model(emb_cfg)
        embedding_device = emb_cfg.get('device') or 'cpu'
        embedding_model_path = emb_path
        embedding_model_name = emb_name
        embedding_use_local = emb_use_local
    else:
        base_path = args.storage
        embedding_model_path = args.embedding_model_path
        embedding_model_name = args.embedding_model_name
        embedding_device = args.embedding_device
        embedding_use_local = args.embedding_use_local and not args.embedding_use_hf

    storage_path = str(Path(base_path) / args.graph_id)

    if not Path(storage_path).exists():
        _log_error("System", f"图谱路径不存在: {storage_path}")
        _log_error("System", f"  基础目录: {base_path}，graph_id: {args.graph_id}")
        return 1

    # Read port from config if --config is used
    if args.config:
        port = config.get('port', args.port)
    else:
        port = args.port

    # Read Neo4j connection info from config
    if args.config:
        neo4j_cfg = (config.get('storage') or {}).get('neo4j') or {}
        neo4j_uri = neo4j_cfg.get('uri', 'bolt://localhost:7687')
        neo4j_user = neo4j_cfg.get('user', 'neo4j')
        neo4j_password = neo4j_cfg.get('password', 'password')
    else:
        neo4j_uri = 'bolt://localhost:7687'
        neo4j_user = 'neo4j'
        neo4j_password = 'password'

    server = GraphWebServer(
        storage_path=storage_path,
        port=port,
        embedding_model_path=embedding_model_path,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
        embedding_use_local=embedding_use_local,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )
    server.run(debug=args.debug, host=args.host)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
