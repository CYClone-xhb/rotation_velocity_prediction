import sys
import random
import numpy as np
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLineEdit, QLabel, QPushButton, QGridLayout,
                            QTabWidget, QFileDialog, QMessageBox, QProgressBar,
                            QDialog, QTableWidget, QTableWidgetItem,
                            QDialogButtonBox, QGroupBox, QFormLayout)
from PyQt5.QtGui import QFont
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import MinMaxScaler
import math
from collections import deque
from PyQt5.QtCore import QMetaObject, Qt, QThread, pyqtSignal, Q_ARG
from matplotlib import font_manager

def config_chinese_font():
    """深度修复matplotlib中文显示问题"""
    from matplotlib import font_manager
    try:
        # Windows系统常见中文字体
        win_fonts = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi']
        # Linux/Mac系统常见中文字体
        unix_fonts = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']

        # 检测系统可用中文字体
        system_fonts = [f.name for f in font_manager.fontManager.ttflist if
                        any(font in f.name for font in win_fonts + unix_fonts)]

        if system_fonts:
            # 取第一个检测到的中文字体
            plt.rcParams['font.sans-serif'] = [system_fonts[0]] + win_fonts + unix_fonts
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已启用中文字体: {system_fonts[0]}")
        else:
            print("⛔ 未检测到中文字体，请安装中文字体文件")
    except Exception as e:
        print(f"字体配置异常: {str(e)}")

# 加载训练好的 CatBoost 模型
catboost_model = CatBoostRegressor()
catboost_model.load_model("catboost_model.cbm")  # 替换为你的模型路径

class DataProcessor:
    def __init__(self, data_path="data/zong-0.csv"):
        self.data = pd.read_csv(data_path)
        self.variable_columns = ['D', 'J', 'M', 'd', 'd1', 'd2',
                                'h', 'h1', 'h2', 'q', 's', 'z']
        self.target_column = 'y'
        self._preprocess_data()

    def _preprocess_data(self):
        """预处理时保持特征顺序"""
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data[self.variable_columns])
        # 必须强制设置scaler的特征名称
        self.scaler.feature_names_in_ = np.array(self.variable_columns)

    def get_variable_stats(self):
        """返回详细的参数统计信息"""
        return {
            col: {
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'dtype': self.data[col].dtype
            } for col in self.variable_columns
        }

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class GAOptimizer(QThread):
    update_progress = pyqtSignal(float)
    optimization_progress = pyqtSignal(dict)  # 新增中间信号
    optimization_finished = pyqtSignal(list)

    def __init__(self, model, scaler, stats, fixed_params, target):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.stats = stats
        self.fixed_params = fixed_params  # 锁定的参数 (d,M,z)
        self.target = target

        # 遗传算法参数配置
        self.population_size = 30
        self.generations = 50
        self.current_gen = 0

        # 自定义参数范围（单位需与训练数据一致）
        self.opt_params = {
            'D': (20, 500),  # 公称直径 (mm)
            'J': (9, 20),  # 锥角 (°)
            'd1': (10, 300),  # 溢流口直径 (mm)
            'd2': (10, 300),  # 底流口直径 (mm)
            'h': (20, 400),  # 插入深度 (mm)
            'h1': (40, 1500),  # 柱段长度 (mm)
            'h2': (40, 1500),  # 锥段长度 (mm)
            'q': (9, 20),  # 进口气速 (m/s)
            's': (90, 60000),  # 进口面积 (mm²)

        }

    def create_individual(self):
        """根据参数范围生成个体"""
        return [
            random.uniform(low, high)
            for (low, high) in self.opt_params.values()
        ]

    def evaluate(self, individual):
        """适应度评估函数"""
        try:


            # 重组完整参数向量
            params = {}
            for i, key in enumerate(self.opt_params.keys()):
                params[key] = individual[i]
            params.update(self.fixed_params)

            # 添加边界校验
            for i, (name, value) in enumerate(zip(self.opt_params.keys(), individual)):
                if not (self.opt_params[name][0] <= value <= self.opt_params[name][1]):
                    return (float('inf'),)  # 直接返回无限误差

            # 按特征顺序生成输入
            ordered = []
            for col in self.scaler.feature_names_in_:
                val = params.get(col, self.stats[col]['min'])
                min_val = self.stats[col]['min']
                max_val = self.stats[col]['max']
                # 归一化
                scaled = (val - min_val) / (max_val - min_val + 1e-8)
                ordered.append(scaled)

            # 反归一化预测
            actual = self.scaler.inverse_transform([ordered])[0]
            pred = self.model.predict([actual])[0]
            error = (pred - self.target) ** 2

            # 参数约束惩罚
            penalty = 0
            NEED_CHECK = ['J', 'd', 'M']  # 需要特别校验的敏感参数

            for i, (name, value) in enumerate(zip(self.opt_params.keys(), individual)):
                low, high = self.opt_params[name]

                # 常规范围校验
                if value < low or value > high:
                    penalty += 1e6 * (abs(value - (low + high) / 2)) ** 2

                # 单位敏感度优化（示例）
                if name in NEED_CHECK:
                    if name == 'J' and value > 30:  # 锥角最大容忍到30度
                        penalty += 1e8
                    elif name == 'd' and value > 10:  # 颗粒直径限制在10mm以内
                        penalty += 1e8

            return error + penalty,

        except Exception as e:
            print(f"评估失败: {str(e)}")
            return float('inf'),

    def run(self):
        try:

            # 初始化统计记录器
            self.gen_stats = {
                'best_fit': [],
                'avg_fit': [],
                'min_fit': [],
                'param_history': {k: [] for k in self.opt_params}
            }

            toolbox = base.Toolbox()
            toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.evaluate)
            toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                             low=[p[0] for p in self.opt_params.values()],
                             up=[p[1] for p in self.opt_params.values()], eta=20)
            toolbox.register("mutate", tools.mutPolynomialBounded,
                             low=[p[0] for p in self.opt_params.values()],
                             up=[p[1] for p in self.opt_params.values()], eta=20, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # 初始化种群
            pop = toolbox.population(n=self.population_size)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)

            # 进化循环
            for gen in range(self.generations):
                # 评估个体前必须清除无效适应度
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

                # 处理异常评估结果
                for ind, fit in zip(invalid_ind, fitnesses):
                    try:
                        ind.fitness.values = fit
                    except:
                        ind.fitness.values = (float('inf'),)  # 给失效个体极大惩罚值

                # 安全生成fits列表
                fits = []
                for ind in pop:
                    try:
                        val = ind.fitness.values[0]
                    except (IndexError, AttributeError):
                        val = float('inf')
                    # 追加边界校验
                    if not math.isfinite(val):
                        val = float('inf')
                    fits.append(val)

                # 选择下一代
                offspring = toolbox.select(pop, len(pop))
                # 克隆选择的个体
                offspring = list(map(toolbox.clone, offspring))

                # 交叉和变异
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.9:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < 0.2:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # 更新种群
                pop[:] = offspring
                hof.update(pop)

                # 更新进度
                self.current_gen = gen + 1
                self.update_progress.emit((self.current_gen / self.generations) * 100)

                # 记录本代特征数据
                best_params = dict(zip(self.opt_params.keys(), hof[0]))
                for k in self.opt_params:
                    self.gen_stats['param_history'][k].append(best_params[k])

                #
                # 收集统计信息
                self.gen_stats['best_fit'].append(min(fits))
                self.gen_stats['avg_fit'].append(np.nanmean(fits))  # 防止空值导致的崩溃
                self.gen_stats['min_fit'].append(np.min(fits))

                # 每秒发射一次更新信号（可选实时更新）
                # 每5代输出调试信息
                if gen % 5 == 0:
                    print(
                        f"代数: {gen + 1} | "
                        f"最佳适应度: {min(fits):.2f} | "
                        f"平均适应度: {np.mean(fits):.2f}"
                    )
                if gen % 1 == 0:
                    self.optimization_progress.emit({
                        'gen': gen,
                        'best': self.gen_stats['best_fit'][-1],
                    })



            # 处理最优解
            best = hof[0]
            best_params = dict(zip(self.opt_params.keys(), best))
            best_params.update(self.fixed_params)

            # 转换完整参数顺序
            ordered_params = []
            for col in self.scaler.feature_names_in_:
                ordered_params.append(best_params[col])

            # 最终预测验证
            scaled = (np.array(ordered_params) - self.scaler.data_min_) / (self.scaler.data_range_ + 1e-8)
            pred = self.model.predict([self.scaler.inverse_transform([scaled])[0]])[0]

            self.optimization_finished.emit([{
                'params': ordered_params,
                'error': abs(pred - self.target),
                'pred': pred,
                'stats': self.gen_stats,  # 新增统计数据字段
                'message': f"完成{self.generations}代进化"
            }])

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.optimization_finished.emit([{
                'error': float('inf'),
                'message': f"优化失败: {str(e)}",
                'pred': 0,  # 新增默认值
                'params': [],  # 新增默认值
                'stats': {}  # 新增默认值
            }])





# 在程序启动前调用
config_chinese_font()




class SymbolicRegressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.forward_inputs = None
        self.inverse_inputs = None
        self.variable_order = ["D", "J", "M", "d", "d1", "d2", "h", "h1", "h2", "q", "s", "z"]
        self.locked_params = {}
        self.optim_figures = {}  # 存储多个可视化图表
        self.evolution_fig = plt.figure()
        self.evolution_canvas = FigureCanvas(self.evolution_fig)
        self.evolution_canvas.setStyleSheet("border: 1px solid #999;")

        self._optimization_results = None  # 新增优化结果缓存
        self._config_plot_style()

        # 修正区域：添加data_processor初始化
        self.data_processor = DataProcessor()  # 新增这行代码
        self.dataset_stats = self.data_processor.get_variable_stats()  # 获取统计信息

        # 检查scaler是否已准备就绪
        if not hasattr(self.data_processor, 'scaler'):
            raise RuntimeError("数据预处理器未正确初始化")

        self.feature_order = self.data_processor.scaler.feature_names_in_.tolist()
        self.initUI()

    def _config_plot_style(self):
        """修正后的绘图参数配置"""
        try:
            # 尝试使用seaborn风格 (需要安装seaborn)
            import seaborn as sns
            sns.set_theme(style="darkgrid")
        except ImportError:
            # 备用方案：使用matplotlib内置风格
            plt.style.use('ggplot')  # 或其他可用风格如'seaborn', 'classic'等

        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 100
        })
    def setup_result_panel(self, layout):
        """结果展示面板配置"""
        # 仅保留进化曲线画布
        self.result_container = QWidget()
        container_layout = QVBoxLayout(self.result_container)

        self.result_display = QLabel("Waiting for calculation results...")
        container_layout.addWidget(self.result_display)

        # 可视化区域只保留进化曲线
        self.visual_area = QVBoxLayout()
        self.visual_area.addWidget(self.evolution_canvas)
        container_layout.addLayout(self.visual_area, stretch=5)

        # 进度条
        self.progress_bar = QProgressBar()
        container_layout.addWidget(self.progress_bar)

        # 添加到主布局
        layout.addWidget(self.result_container)

    def initUI(self):
        self.setWindowTitle("Cyclone particle rotation velocity prediction software 1.0")
        self.setFixedSize(1200, 800)  # 扩大窗口尺寸

        # 主布局
        main_layout = QHBoxLayout()

        # 左侧布局
        left_panel = QVBoxLayout()

        # 创建选项卡
        self.tabs = QTabWidget()
        self.add_forward_tab()  # 正向预测
        self.add_inverse_tab()  # 新增反向计算
        left_panel.addWidget(self.tabs)

        # 公共控制按钮
        control_layout = QHBoxLayout()
        btn_style = "QPushButton {padding: 8px 15px; font-size:14px;}"

        self.export_btn = QPushButton("Export results")
        self.export_btn.setStyleSheet(btn_style)
        self.export_btn.clicked.connect(self.export_data)

        self.clear_btn = QPushButton("Clear Input")
        self.clear_btn.setStyleSheet(btn_style)
        self.clear_btn.clicked.connect(self.clear_current_tab)

        control_layout.addWidget(self.export_btn)
        control_layout.addWidget(self.clear_btn)
        left_panel.addLayout(control_layout)

        # 右侧布局
        right_panel = QVBoxLayout()
        self.setup_result_panel(right_panel)  # 结果展示面板

        main_layout.addLayout(left_panel, 3)  # 左3右7比例
        main_layout.addLayout(right_panel, 7)
        self.setLayout(main_layout)

    def add_forward_tab(self):
        """正向预测选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建输入网格并获取布局和输入字典
        grid, self.forward_inputs = self.create_input_grid("forward")
        layout.addLayout(grid)  # 只添加布局，不是元组

        # 预测按钮
        self.predict_btn = QPushButton("prediction")
        self.predict_btn.setStyleSheet("QPushButton {background: #4CAF50; color: white;}")
        self.predict_btn.clicked.connect(self.run_prediction)
        layout.addWidget(self.predict_btn)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "forward prediction")

    # 主要修改位置：SymbolicRegressionApp类的add_inverse_tab方法
    def add_inverse_tab(self):
        """强化反向计算输入界面"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建参数输入网格
        grid = QGridLayout()
        self.inverse_inputs = {}


        # 参数配置字典（优化显示顺序）
        param_config = [
            ("y", ("目标转速", "rad/s")),
            ("M", ("颗粒密度", "kg/m³")),
            ("d", ("颗粒直径", "mm")),
            ("z", ("位置", "")),
            ("D", ("公称直径", "mm")),
            ("J", ("锥角", "°")),
            ("d1", ("溢流口直径", "mm")),
            ("d2", ("底流口直径", "mm")),
            ("h", ("插入深度", "mm")),
            ("h1", ("柱段长度", "mm")),
            ("h2", ("锥段长度", "mm")),
            ("q", ("进口气速", "m/s")),
            ("s", ("进口面积", "mm²"))
        ]

        # 生成输入行
        for row, (var, (name, unit)) in enumerate(param_config):
            label = QLabel(f"{name}:")
            input_box = QLineEdit()
            grid.addWidget(label, row, 0)
            grid.addWidget(input_box, row, 1)
            grid.addWidget(QLabel(unit), row, 2)
            self.inverse_inputs[var] = input_box


        layout.addLayout(grid)
        self._add_optimize_button(layout)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "") # 反向计算


    def _add_target_input(self, layout, name, unit):
        """添加目标转速输入行"""
        target_layout = QHBoxLayout()
        target_label = QLabel(f"{name}:")
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText(f"输入{name}")

        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_input)
        target_layout.addWidget(QLabel(unit))
        layout.addLayout(target_layout)

    def _add_optimize_button(self, layout):
        """添加优化操作按钮"""
        self.inverse_btn = QPushButton("启动反算优化")
        self.inverse_btn.setObjectName("optimizeButton")
        self.inverse_btn.setStyleSheet("""
            #optimizeButton {
                background: #2196F3;
                color: white;
                padding: 12px;
                font-size: 14px;
                border-radius: 4px;
            }
            #optimizeButton:disabled {
                background: #BBDEFB;
            }
        """)
        self.inverse_btn.clicked.connect(self.validate_inverse_inputs)
        layout.addWidget(self.inverse_btn)

    def validate_inverse_inputs(self):
        """反向计算输入验证"""
        errors = []

        # 初始化参数容器
        validated_params = {
            'target': None,
            'd': None,
            'M': None,
            'z': None
        }

        # 检查目标转速 (对应输入框的key是"y")
        target_input = self.inverse_inputs["y"].text().strip()
        if not target_input:
            errors.append("必须输入目标转速")
        else:
            try:
                validated_params['target'] = float(target_input)
                if validated_params['target'] <= 0:
                    errors.append("目标转速必须为正数")
            except ValueError:
                errors.append("目标转速格式无效")

        # 检查颗粒直径 (key: "d")
        d_input = self.inverse_inputs["d"].text().strip()
        if not d_input:
            errors.append("必须输入颗粒直径")
        else:
            try:
                validated_params['d'] = float(d_input)
                if validated_params['d'] <= 0:
                    errors.append("颗粒直径必须为正数")
            except ValueError:
                errors.append("颗粒直径格式无效")

        # 检查颗粒密度 (key: "M")
        M_input = self.inverse_inputs["M"].text().strip()
        if not M_input:
            errors.append("必须输入颗粒密度")
        else:
            try:
                validated_params['M'] = float(M_input)
                if validated_params['M'] <= 0:
                    errors.append("颗粒密度必须为正数")
            except ValueError:
                errors.append("颗粒密度格式无效")

        # 检查位置 (key: "z")
        z_input = self.inverse_inputs["z"].text().strip()
        if not z_input:
            errors.append("必须输入位置")
        else:
            try:
                validated_params['z'] = float(z_input)
            except ValueError:
                errors.append("位置格式无效")

        # 处理验证结果
        if errors:
            self.show_error_message("验证错误：\n• " + "\n• ".join(errors))
        else:
            # 二次校验参数完整性
            missing_params = [k for k, v in validated_params.items() if v is None]
            if missing_params:
                self.show_error_message(f"参数完整性错误: {missing_params}")
                return

            # 确保数值有效性
            try:
                self.locked_params = {
                    'd': validated_params['d'],
                    'M': validated_params['M'],
                    'z': validated_params['z']
                }
                self.start_optimization(validated_params['target'])
            except Exception as e:
                self.show_error_message(f"参数传递错误: {str(e)}")

    def start_optimization(self, target):
        """启动遗传算法优化"""
        try:
            self.progress_bar.setValue(0)
            self.optimizer = GAOptimizer(
                model=catboost_model,
                scaler=self.data_processor.scaler,
                stats=self.dataset_stats,
                fixed_params=self.locked_params,
                target=target
            )
            self.optimizer.update_progress.connect(self.update_progress)
            self.optimizer.optimization_finished.connect(self.show_optimization_results)
            self.optimizer.start()
            self.inverse_btn.setEnabled(False)

        except Exception as e:
            self.show_error_message(f"优化器初始化失败: {str(e)}")
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(int(value))

    def show_optimization_results(self, results):
        """显示优化结果（修复完整版）"""
        try:
            # ==== 数据校验 ====
            if not results or not isinstance(results, list):
                self.result_display.setText("优化结果格式错误")
                return

            res = results[0]
            if 'stats' not in res:
                self.result_display.setText("缺少统计信息")
                return

            stats = res['stats']
            if not stats.get('best_fit'):
                self.result_display.setText("无进化轨迹数据")
                return

            # ==== 生成绘图数据 ====
            gens = list(range(1, len(stats['best_fit']) + 1))

            # ==== 清理画布 ====
            self.evolution_fig.clf()
            self.evolution_canvas.setVisible(True)  # 确保画布可见

            # ==== 新增调试信息 ====
            print(f"正在绘制曲线，代数数量: {len(gens)}, 数据点数量: {len(stats['best_fit'])}")
            ax = self.evolution_fig.add_subplot(111)
            ax.clear()

            # ==== 绘制双曲线 ====
            line_best, = ax.plot(gens, stats['best_fit'],
                                 color='#2196F3',
                                 linewidth=2.5,
                                 marker='o',
                                 markersize=4,
                                 label='最佳适应度')

            line_avg, = ax.plot(gens, stats['avg_fit'],
                                color='#FF5722',
                                linestyle='--',
                                linewidth=1.5,
                                label='平均适应度')

            # ==== 强制调整布局 ====
            self.evolution_fig.tight_layout()
            self.evolution_canvas.draw()  # 强制执行绘制

            # ==== 立即刷新界面 ====
            QApplication.processEvents()  # 强制处理所有pending事件

            # ==== 中文元素配置 ====
            ax.set_title("遗传算法优化过程",
                         fontsize=14,
                         fontweight='bold',
                         pad=12)
            ax.set_xlabel("迭代代数", fontsize=12, labelpad=8)
            ax.set_ylabel("适应度值", fontsize=12, labelpad=8)

            # ==== 图例配置 ====
            legend = ax.legend(
                handles=[line_best, line_avg],
                loc='upper right',
                frameon=True,
                framealpha=0.9,
                edgecolor='#CCCCCC',
                facecolor='#FFFFFF',
                prop={
                    'family': plt.rcParams['font.sans-serif'][0],
                    'size': 10
                }
            )

            # ==== 动态范围调整 ====
            valid_values = [v for v in stats['best_fit'] if v != float('inf')]
            if len(valid_values) < 2:
                ax.text(0.5, 0.5, '有效数据不足',
                        ha='center', va='center',
                        fontsize=12, color='gray')
                return

            y_min = min(valid_values) * 0.98
            y_max = max(valid_values) * 1.02
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(1, len(gens))

            # ==== 网格和刻度 ====
            ax.grid(True,
                    axis='both',
                    linestyle='--',
                    alpha=0.4)
            ax.tick_params(axis='both',
                           which='both',
                           labelsize=10)

            # ==== 实时渲染 ====
            self.evolution_canvas.draw_idle()
            self.evolution_canvas.flush_events()

            # ==== 更新文本结果 ====
            if res['error'] < 1e6:
                self._update_result_text(res)
            else:
                self.result_display.setText("未找到可行解")

        except Exception as e:
            self.result_display.setText(f"结果渲染失败: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"当前字体配置: {plt.rcParams['font.sans-serif']}")
        finally:
            # ==== 关键修复代码 =====
            self.inverse_btn.setEnabled(True)  # 无论成功失败都恢复按钮
            self.progress_bar.setValue(100)  # 可选：确保进度条完成

    def _update_result_text(self, res):
        """更新文字结果显示（完整版）"""
        # ===== 参数加载 =====
        try:
            # 将优化参数逆标准化（转回原始量纲）
            scaled_params = np.array(res['params'])
            actual_params = self.data_processor.scaler.inverse_transform([scaled_params])[0]

            # 与特征名称组合成字典
            params_dict = {}
            for i, col in enumerate(self.feature_order):
                params_dict[col] = actual_params[i]

                # 将计算后的参数设置到输入框（跳过锁定的参数）
                if col not in self.locked_params:
                    input_box = self.inverse_inputs.get(col)
                    if input_box:
                        input_box.setText(f"{actual_params[i]:.2f}")

        except Exception as e:
            self.result_display.setText(f"参数转换失败: {str(e)}")
            return

        # ===== 构建显示文本 =====
        result_text = (
            f"目标转速: {self.optimizer.target:.2f} rad/s\n"
            f"预测结果: {res['pred']:.2f} rad/s\n"
            f"绝对误差: {res['error']:.2f}\n\n"
            "优化参数：\n"
        )

        # 添加优化的参数信息
        param_details = []
        for col in self.feature_order:
            if col in self.locked_params:
                continue  # 跳过锁定的参数

            chinese_name = self.vars_to_chinese([col])[0]
            unit = self.get_unit(col)
            value = params_dict.get(col, 0.0)

            param_details.append(
                f"{chinese_name}: {value:.2f}{unit}"
            )

        # 组合最终显示内容
        full_text = result_text + "\n".join(param_details)
        self.result_display.setText(full_text)

    def _render_graphics(self, results):
        if not hasattr(self, '_optimization_results') or not self._optimization_results:
            print("无有效结果可渲染")
            return

        if 'stats' not in self._optimization_results[0]:
            self.result_display.setText("优化结果格式错误")
            return
        """真正执行绘图的核心方法，确保在主线程运行"""
        if not self._optimization_results:
            return
        res = results[0]
        stats = res.get('stats', {})

        # ==== 1. 绘制进化曲线 ====
        self.evolution_fig.clf()
        ax1 = self.evolution_fig.add_subplot(111)
        if len(stats.get('best_fit', [])) > 2:
            gens = range(1, len(stats['best_fit']) + 1)
            ax1.plot(gens, stats['best_fit'], 'r-', label='最佳适应度')
            ax1.set_title("遗传算法进化曲线")
        else:
            ax1.text(0.5, 0.5, '数据点不足', ha='center')
        self.evolution_canvas.draw()

        # ==== 2. 绘制雷达图 ====
        self.radar_fig.clf()
        ax2 = self.radar_fig.add_subplot(111, polar=True)
        params = stats.get('param_history', {})

        labels, values = [], []
        for k in params.keys():
            if k in self.optimizer.opt_params:
                low, high = self.optimizer.opt_params[k]
                mean_val = np.mean(params[k]) if len(params[k]) > 0 else low
                labels.append(self.vars_to_chinese([k])[0])
                values.append((mean_val - low) / (high - low + 1e-8))

        if values:
            values += values[:1]
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]
            ax2.plot(angles, values, 'b-')
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(labels)
        else:
            ax2.text(0.5, 0.5, '无参数轨迹', ha='center')
        self.radar_canvas.draw()  # 同步绘制确保立即更新

        # ==== 3. 绘制条形图 ====
        self.bar_fig.clf()
        ax3 = self.bar_fig.add_subplot(111)
        bars = ax3.bar(
            ['目标值', '预测值'],
            [self.optimizer.target, res.get('pred', 0)],
            color=['#FF5722', '#607D8B']
        )
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center')
        self.bar_canvas.draw()


    def load_optimized_params(self, params):
        """加载优化参数到输入框"""
        for var, value in zip(self.variable_order, params):
            if var not in self.locked_params:
                self.inverse_inputs[var].setText(f"{value:.2f}")

    def load_selected_params(self, table):
        """加载选中参数到界面"""
        row = table.currentRow()
        params = self.results_cache[row]['params']

        for var, value in zip(self.variable_order, params):
            if var in self.locked_params:
                continue
            self.inverse_inputs[var].setText(f"{value:.2f}")

    def is_valid_number(self, text):
        """验证输入是否为合法数字（支持整数和浮点数）"""
        pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
        return re.match(pattern, text) is not None

    def create_input_grid(self, mode):
        variables = self.variable_order
        # meanings = {
        #     "D": ("公称直径", "mm"), "h": ("插入深度", "mm"),
        #     "d1": ("溢流口直径", "mm"), "d2": ("底流口直径", "mm"),
        #     "h1": ("柱段长度", "mm"), "h2": ("锥段长度", "mm"),
        #     "J": ("锥角", "°"), "s": ("进口面积", "mm²"),
        #     "d": ("颗粒直径", "mm"), "M": ("颗粒密度", "kg/m³"),
        #     "q": ("进口气速", "m/s"), "z": ("位置", "")
        # }

        meanings = {
            "D": ("Cyclone diameter", "mm"), "h": ("insert height", "mm"),
            "d1": ("Overflow diameter", "mm"), "d2": ("Lowflow diameter", "mm"),
            "h1": ("Column height", "mm"), "h2": ("Cone height", "mm"),
            "J": ("Cone angle", "°"), "s": ("Inlet area", "mm²"),
            "d": ("Particle diameter", "mm"), "M": ("Particle density", "kg/m³"),
            "q": ("Inlet velocity", "m/s"), "z": ("Z", "")
        }

        grid = QGridLayout()
        inputs = {}

        for i, var in enumerate(variables):
            label = QLabel(f"{meanings[var][0]}:")
            input_box = QLineEdit()

            # 反向计算参数标识
            if mode == "inverse":
                if var in ["D", "J", "d1", "d2", "h", "h1", "h2", "q", "s"]:
                    input_box.setPlaceholderText("反算参数")
                    input_box.setStyleSheet("background: #FFF3E0;")
                elif var in ["M", "d","z"]:  # 示例：锁定部分参数
                    input_box.setReadOnly(True)
                    input_box.setStyleSheet("background: #EEE;")

            grid.addWidget(label, i, 0)
            grid.addWidget(input_box, i, 1)

            # 单位标签
            unit_label = QLabel(meanings[var][1])
            grid.addWidget(unit_label, i, 2)
            inputs[var] = input_box

            # 仅反向计算显示目标输入
        if mode == "inverse":
            target_row = len(variables)
            grid.addWidget(QLabel("目标转速:"), target_row, 0)
            target_input = QLineEdit()
            grid.addWidget(target_input, target_row, 1)
            grid.addWidget(QLabel("rad/s"), target_row, 2)
            inputs["target"] = target_input

        return grid, inputs

    def setup_result_panel(self, layout):
        """配置右侧结果展示区"""
        # 结果文字显示
        self.result_display = QLabel("Waiting for calculation results...")
        self.result_display.setStyleSheet("""
            QLabel {
                font-size: 14px; 
                padding: 15px;
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                min-height: 100px;
            }
        """)
        self.result_display.setWordWrap(True)

        # 可视化图表区域
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                height: 25px;
                text-align: center;
                border: 1px solid #BDBDBD;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)

        layout.addWidget(self.result_display)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.progress_bar)

    def run_prediction(self):
        """执行正向预测流程"""
        try:
            # 初始化进度条
            self.show_progress(0)

            # 按模型输入顺序收集参数
            inputs = self.forward_inputs  # 获取正确的inputs字典
            input_data = [inputs[var].text().strip() for var in self.variable_order]

            # 输入校验
            errors = []
            valid_values = []
            for var, val in zip(self.variable_order, input_data):
                if not val:
                    errors.append(f"缺失参数：{self.vars_to_chinese([var])[0]}")
                    continue

                try:
                    num_val = float(val)
                    # 参数特异性校验
                    if var == "M" and num_val <= 0:
                        raise ValueError("颗粒密度必须大于0")
                    elif var == "d" and num_val < 0:
                        raise ValueError("颗粒直径不能为负")
                    elif var == "q" and num_val < 0:
                        raise ValueError("进口气速不能为负")
                    elif var in ["D", "d1", "d2", "h", "h1", "h2"] and num_val <= 0:
                        raise ValueError("几何尺寸必须大于0")
                    elif var == "J" and not (0 <= num_val <= 50):
                        raise ValueError("锥角需在0-50度之间")

                    valid_values.append(num_val)
                except ValueError as e:
                    errors.append(f"{self.vars_to_chinese([var])[0]}：{str(e)}")

            # 立即显示首个错误
            if errors:
                self.show_error_message("• " + "\n• ".join(errors[:5]))  # 最多显示前3个错误
                return

            # 执行预测
            self.show_progress(30)
            prediction = catboost_model.predict([valid_values])[0]

            # 结果展示优化
            params_text = "\n".join([
                f"• {self.vars_to_chinese([var])[0]}: {val}{self.get_unit(var)}"
                for var, val in zip(self.variable_order, valid_values)
            ])
            self.result_display.setText(
                f"【Success】\n"
                f"Rotation velocity：{prediction:.2f} rad/s\n\n"
                f"【Input Parameters】\n{params_text}"
            )

            # 动态可视化
            self.show_progress(60)
            self.update_visualization(valid_values, prediction)
            self.show_progress(100)

        except Exception as e:
            self.result_display.setText(f"预测异常：{str(e)}")
            self.show_progress(0)

    def show_progress(self, value):
        """新增进度条方法"""
        self.progress_bar.setValue(value)
    def show_error_message(self, message):
            """优化的错误提示方法"""
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("输入验证失败")


            # 自动去除重复变量代号的括号内容
            clean_msg = re.sub(r"$[^)]+$", "", message)
            msg.setText(clean_msg)

            # 添加详细解释
            msg.setInformativeText("请按以下要求修正输入：\n"
                                   "- 红色标注参数必须填写\n"
                                   "- 数值需符合物理意义范围")
            msg.exec_()

    def get_unit(self, var):
            """带范围说明的单位标签"""
            units = {

                "D": (" mm",""),
                "h": (" mm",""),
                "d1": (" mm",""),
                "d2": (" mm",""),
                "h1": (" mm",""),
                "h2": (" mm",""),
                "J": ("°",""),
                "s": (" mm²",""),
                "d": (" mm",""),
                "M": (" kg/m³",""),
                "q": (" m/s",""),
            }
            unit, range_desc = units.get(var, ("", ""))
            return f"{unit} ({range_desc})" if range_desc else unit

    def vars_to_chinese(self, variables):
        """将变量代号转换为中文"""
        translation = {
            "D": "D", "J": "J", "M": "M",
            "d": "d", "d1": "d1", "d2": "d2",
            "h": "h", "h1": "h1", "h2": "h2",
            "q": "q", "s": "s", "z": "z"
        }
        return ", ".join([translation.get(var, var) for var in variables])

    def update_visualization(self, input_data, prediction):
        """修复后的可视化图表（解决中文显示和标注定位）"""
        try:
            # ==== 字体配置 ====
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 多字体备选
            plt.rcParams['axes.unicode_minus'] = False

            # ==== 初始化画布 ====
            self.figure.clf()
            ax = self.figure.add_subplot(111)

            # ==== 数据准备 ====
            importance = catboost_model.get_feature_importance()
            features = ["D", "J", "M", "d", "d1", "d2", "h", "h1", "h2", "q", "s", "z"]
            sorted_idx = np.argsort(importance)
            sorted_importance = importance[sorted_idx]
            chinese_features = [self.vars_to_chinese([f])[0] for f in np.array(features)[sorted_idx]]

            # ==== 解决参数名显示不全 ====
            ax.tick_params(axis='y', labelrotation=0, labelsize=4)  # 确保y轴标签水平显示
            plt.subplots_adjust(left=0.6)  # 关键：给y轴标签留出足够空间

            # ==== 绘制条形图 ====
            y_pos = np.arange(len(chinese_features))
            bars = ax.barh(
                y_pos,
                sorted_importance,
                height=0.6,  # 调小条带高度防止重叠
                color='#4CAF50',
                edgecolor='black',
                alpha=0.8
            )

            # ==== 设置标签 ====
            ax.set_yticks(y_pos)
            ax.set_yticklabels(chinese_features, fontsize=12, va='center')  # 保证垂直居中
            ax.set_title("Parameter influence weighting analysis", fontsize=14, pad=20)
            ax.set_xlabel("Coefficient of importance of the feature", fontsize=12)

            # ==== 解决数值标注偏移问题 ====
            max_importance = max(sorted_importance)
            for i, (pos, imp) in enumerate(zip(y_pos, sorted_importance)):
                ax.text(
                    imp + max_importance * 0.02,  # 右偏移量基于最大值为基准
                    pos,  # y坐标对准条带中心
                    f'{imp:.2f}',
                    va='center',  # 垂直居中
                    ha='left',  # 左对齐文本
                    fontsize=10,
                    color='#333'
                )

            # ==== 布局优化 ====
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()

            self.canvas.draw()

        except Exception as e:
            print(f"可视化失败：{str(e)}")
            # 初始化错误占位图表
            self.figure.clf()
            ax_fail = self.figure.add_subplot(111)
            ax_fail.text(
                0.5, 0.5,
                '图表初始化失败\n请检查字体配置',
                ha='center',
                va='center',
                fontsize=14,
                fontproperties='SimHei'
            )
            self.canvas.draw()

    def export_data(self):
        """统一导出处理：CSV数据 + TIFF图像"""
        try:
            # 选择保存路径
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存结果", "",
                "CSV Files (*.csv);;TIFF Images (*.tiff)"
            )
            if not save_path:
                return  # 用户取消保存

            # 分页面处理
            current_tab = self.tabs.currentIndex()
            current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

            # 处理正向预测标签页
            if current_tab == 0:
                # 收集输入参数
                params_data = {
                    self.vars_to_chinese([var])[0]: float(input_box.text())
                    for var, input_box in self.forward_inputs.items()
                    if input_box.text().strip()
                }
                # 读取预测结果
                result_match = re.search(r"自转速度：(\d+\.?\d*)", self.result_display.text())
                if result_match:
                    params_data["预测值(rad/s)"] = float(result_match.group(1))

                df = pd.DataFrame([params_data])
                csv_path = f"{save_path.rsplit('.', 1)[0]}_{current_time}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8_sig')

            # 处理反向计算标签页
            elif current_tab == 1:
                if not hasattr(self, '_optimization_results') or not self._optimization_results:
                    raise ValueError("请先完成优化计算")

                res = self._optimization_results[0]
                stats = res['stats']

                # 导出优化参数
                params = self.data_processor.scaler.inverse_transform([res['params']])[0]
                param_df = pd.DataFrame({
                    '变量': self.feature_order,
                    '原始量纲值': params,
                    '归一化值': res['params']
                })

                # 附加优化信息
                info_df = pd.DataFrame({
                    '优化目标(rad/s)': [self.optimizer.target],
                    '实际预测值(rad/s)': [res['pred']],
                    '绝对误差': [res['error']],
                    '优化用时(代)': [len(stats['best_fit'])]
                })

                # 合并数据
                full_df = pd.concat([param_df, info_df], axis=1)
                csv_path = f"{save_path.rsplit('.', 1)[0]}_{current_time}.csv"
                full_df.to_csv(csv_path, index=False, encoding='utf-8_sig')

                # 导出进化曲线图像
                if hasattr(self, 'evolution_fig') and self.evolution_fig.axes:
                    img_path = f"{save_path.rsplit('.', 1)[0]}_{current_time}.tiff"
                    self.evolution_fig.savefig(
                        img_path,
                        dpi=300,
                        format='tiff',
                        bbox_inches='tight',
                        pil_kwargs={"compression": "tiff_lzw"}  # 启用LZW压缩
                    )

            # 提示导出成功
            QMessageBox.information(
                self, "导出成功",
                f"文件已保存至:\n{csv_path}" +
                (f"\n{img_path}" if current_tab == 1 else "")
            )

        except Exception as e:
            QMessageBox.critical(
                self, "导出失败",
                f"错误详情:\n{str(e)}\n请确保已完成计算"
            )

    def clear_current_tab(self):
        """清空当前标签页输入"""
        # 清空所有输入框
        if self.tabs.currentIndex() == 0:  # 如果当前是正向预测标签
            for var, input_box in self.forward_inputs.items():
                input_box.clear()
        elif self.tabs.currentIndex() == 1:  # 如果当前是反向计算标签
            for var, input_box in self.inverse_inputs.items():
                input_box.clear()

        # 清空结果显示
        self.result_display.setText("等待计算结果...")

        # 清空进度条
        self.progress_bar.setValue(0)

        # 清空图像
        self.figure.clf()  # 清空画布
        self.canvas.draw()  # 重绘空图像


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SymbolicRegressionApp()
    window.show()
    sys.exit(app.exec_())