# src/utils/paths.py
from pathlib import Path
import yaml


class ProjectPaths:
    """管理项目路径 - 完整版"""

    def __init__(self, project_root=None):
        """
        初始化路径管理器

        Parameters:
        -----------
        project_root : str or Path, optional
            项目根目录，如果为None则自动检测
        """
        if project_root is None:
            # 自动检测项目根目录
            self.root = self._detect_project_root()
        else:
            self.root = Path(project_root)

        self._load_config()

    def _detect_project_root(self):
        """自动检测项目根目录"""
        # 从当前文件位置开始向上查找
        current = Path(__file__).absolute()

        # 如果是src/utils/paths.py，向上3级到项目根目录
        if "src" in current.parts and "utils" in current.parts:
            # 向上找3级：src/utils/.. → src/.. → project_root
            project_root = current.parent.parent.parent
            if project_root.exists():
                return project_root

        # 如果找不到，使用当前工作目录
        return Path.cwd()

    def _load_config(self):
        """加载路径配置"""
        config_path = self.root / "config" / "paths.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

    # ============ 路径属性 ============

    @property
    def data_raw(self):
        """原始数据目录"""
        return self.root / self.config.get('paths', {}).get('data_raw', 'data/raw')

    @property
    def data_processed(self):
        """处理后的数据目录"""
        return self.root / self.config.get('paths', {}).get('data_processed', 'data/processed')

    @property
    def models(self):
        """训练好的模型目录"""
        return self.root / self.config.get('paths', {}).get('models', 'models/trained')

    @property
    def model_metrics(self):
        """模型评估指标目录"""
        return self.root / self.config.get('paths', {}).get('model_metrics', 'models/metrics')

    @property
    def reports_figures(self):
        """报告图表目录"""
        return self.root / self.config.get('paths', {}).get('reports_figures', 'reports/figures')

    @property
    def reports_tables(self):
        """报告表格目录"""
        return self.root / self.config.get('paths', {}).get('reports_tables', 'reports/tables')

    @property
    def src(self):
        """源代码目录"""
        return self.root / self.config.get('paths', {}).get('src', 'src')

    # ============ 数据文件路径 ============

    @property
    def user_behavior_file(self):
        """用户行为数据文件路径"""
        filename = self.config.get('data_files', {}).get('user_behavior', 'user_behavior.csv')
        return self.data_raw / filename

    @property
    def user_action_processed_file(self):
        """处理后的用户行为数据文件路径"""
        filename = self.config.get('data_files', {}).get('user_action_processed', 'user_action_processed.parquet')
        return self.data_processed / filename

    # ============ 预处理配置 ============

    @property
    def preprocessing_config(self):
        """获取数据预处理配置"""
        return self.config.get('data_preprocessing', {})

    @property
    def chunksize(self):
        """分块大小"""
        return self.preprocessing_config.get('chunksize', 100000)

    @property
    def behavior_type_mapping(self):
        """行为类型映射"""
        return self.preprocessing_config.get('behavior_type_mapping', {
            1: "view",
            2: "favorite",
            3: "cart",
            4: "purchase"
        })

    @property
    def deduplication_config(self):
        """去重规则配置"""
        return self.preprocessing_config.get('deduplication', {})

    @property
    def dtype_optimization(self):
        """数据类型优化配置"""
        return self.preprocessing_config.get('dtype_optimization', {})

    @property
    def missing_values_config(self):
        """缺失值处理配置"""
        return self.preprocessing_config.get('missing_values', {})

    # ============ 其他方法 ============

    def create_all_dirs(self):
        """创建所有需要的目录"""
        dirs = [
            self.data_raw,
            self.data_processed,
            self.models,
            self.model_metrics,
            self.reports_figures,
            self.reports_tables,
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {dir_path.relative_to(self.root)}")

        return self

    def get_relative_path(self, path):
        """获取相对于项目根目录的路径"""
        if isinstance(path, str):
            path = Path(path)
        try:
            return path.relative_to(self.root)
        except ValueError:
            return path

    def __str__(self):
        """字符串表示"""
        lines = [
            "=" * 50,
            "项目路径配置",
            "=" * 50,
            f"项目根目录: {self.root}",
            f"原始数据目录: {self.data_raw}",
            f"处理数据目录: {self.data_processed}",
            f"模型目录: {self.models}",
            f"模型指标目录: {self.model_metrics}",
            f"报告图表目录: {self.reports_figures}",
            f"报告表格目录: {self.reports_tables}",
            f"源代码目录: {self.src}",
            "",
            "数据文件:",
            f"  原始数据文件: {self.user_behavior_file}",
            f"  处理数据文件: {self.user_action_processed_file}",
            "",
            "预处理配置:",
            f"  分块大小: {self.chunksize:,}",
            f"  行为类型映射: {self.behavior_type_mapping}",
            f"  去重配置: {self.deduplication_config}",
            "=" * 50
        ]
        return "\n".join(lines)


# 全局路径对象
PATHS = ProjectPaths()

# 测试
if __name__ == "__main__":
    print(PATHS)
    print("\n" + "=" * 50)
    print("目录存在性检查:")
    print(f"原始数据目录是否存在: {PATHS.data_raw.exists()}")
    print(f"处理数据目录是否存在: {PATHS.data_processed.exists()}")
    print(f"原始数据文件是否存在: {PATHS.user_behavior_file.exists()}")

    # 创建目录
    print("\n" + "=" * 50)
    print("创建目录:")
    PATHS.create_all_dirs()