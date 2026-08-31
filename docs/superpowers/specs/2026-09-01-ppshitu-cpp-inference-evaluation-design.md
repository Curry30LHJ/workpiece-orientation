# PP-ShiTuV2 原生 C++ 推理评估设计

日期：2026-09-01  
分支：`feature/20260901/ppshitu-cpp-inference`  
基线提交：`5c600532689e8eb7cfc32217beb7181affe9f0e1`

## 1. 背景与结论边界

当前 CPU 快速路径已经完成三槽位精确去重和 4×1 独立会话池。开发机上的五图批次 P95 仍约为 132 ms，目标工控机 i5-8250U 的五图 P95 目标为 100 ms。

PaddleClas 的 Python `RecPredictor` 最终仍调用 Paddle Inference 的原生预测器。因此，把相同模型调用机械地从 Python 改写成 C++，只会消除 Python 预处理、对象转换和调度开销，不能假定模型计算本身会自动变快。完整重写后端之前，必须用同模型、同图片、同预处理和同线程配置建立原生 C++ 对照证据。

本设计是第一阶段的独立子项目：建立可复现的 Windows x64 C++ 特征提取与性能评估工具，并用硬门槛决定是否进入产品接入。它不承诺在没有实测证据时替换现有后端。

## 2. 目标

1. 基于 PaddleClas 官方 `deploy/cpp_shitu` 特征提取实现和 Paddle Inference C++ API，构建 Windows 10/11 x64、MSVC 2019、CPU AVX/MKL 的原生识别基准程序。
2. 仅运行当前项目实际使用的 PP-ShiTuV2 `general_PPLCNetV2_base_pretrained_v1.0_infer` 特征模型；跳过主体检测和 FAISS 检索。
3. 证明 C++ 与当前 Python 路径在图像预处理、L2 特征归一化、特征维度、预测标签和复核决策上的一致性。
4. 对单图、五图单预测器批量和五图多预测器并行三种模式分别报告预处理、模型、后处理和端到端 P50/P95/P99。
5. 只有在正确性门禁全部通过、同口径五图特征提取 P95 相对当前 Python `RecPredictor` 至少改善 15%，且端到端投影也至少改善 15% 时，才允许后续提出产品接入设计。
6. 保持当前 CPU 分支、现有工件库、几何规则、模板缓存、Ridge 参数和 Qt 行为不变。

## 3. 非目标

- 不重写 Qt 或当前 TCP 后端。
- 不在本阶段将 C++ 可执行文件或 DLL 放入正式离线包。
- 不修改 PP-ShiTu、ALIKED、LightGlue、几何规则、Ridge 参数或融合阈值。
- 不重新训练、微调、量化或替换模型。
- 不引入主体检测；输入仍是已经裁剪、包含单个主要工件的图片。
- 不引入 FAISS；快速路径使用现有缓存和 Ridge，FAISS 对本次瓶颈判断没有帮助。
- 不把开发机结果冒充 i5-8250U 验收结果。

## 4. 方案比较

### 4.1 方案 A：原生 C++ 对照基准后再决定接入（采用）

先实现识别模型的独立 C++ 基准和特征一致性工具。若收益不足，停止产品迁移并保留证据；若收益达到门槛，再单独设计接入边界。

优点是能以最小改动回答真正的性能问题，不影响当前可交付版本，也能区分 Python 调度与模型计算的占比。缺点是第一阶段不会直接生成新的正式后端。

### 4.2 方案 B：立即用 C++ DLL 替换 Python `RecPredictor`

Python 通过 C ABI 调用 DLL。该方案看似改动小，但同一进程可能同时加载 Python Paddle 与独立 Paddle Inference、MKL/oneDNN DLL，存在版本冲突和重复内存问题；同时 Python Paddle 本身已经使用原生预测器，收益未经证明。

本阶段不采用。

### 4.3 方案 C：立即把完整后端重写为 C++

需要迁移工件库事务、缓存恢复、几何规则、Ridge、批量协议、异步入库、错误码和离线包审计。范围大，且主要模型耗时未必因语言变化下降。

本阶段不采用。只有方案 A 达到收益门槛后，才比较“原生特征服务”和“完整原生后端”两种接入方式。

## 5. 架构与文件边界

新增独立目录，不修改 `third_party/PaddleClas` 中的官方源码：

```text
native/ppshitu_rec_benchmark/
  CMakeLists.txt                 Windows/MSVC 构建入口
  include/feature_extractor.h   原生预测器的窄接口
  include/preprocess.h          与 Python 配置一致的预处理接口
  src/feature_extractor.cpp     Paddle Inference 初始化、批量预测、L2 归一化
  src/preprocess.cpp            Unicode 路径解码、BGR->RGB、resize/normalize/CHW
  src/main.cpp                  CLI、预热、计时和 JSON 报告
  NOTICE.md                     说明复用的 PaddleClas Apache-2.0 来源

scripts/
  build_cpp_inference_benchmark.ps1
  compare_cpp_python_inference.py

tests/
  test_cpp_inference_benchmark_contract.py
  test_compare_cpp_python_inference.py
```

职责划分：

- C++ 程序只负责图片解码、预处理、特征提取和计时，不理解工件库和正反面标签。
- Python 对照脚本负责调用当前 `RecPredictor`、运行同一图片清单、比较 embedding，并用现有工件库分类逻辑检查最终标签和复核决策；它还从当前 `predict_batch` 报告中读取非模型阶段耗时，计算接入后的端到端投影。
- 构建脚本只接受显式的 Paddle Inference、OpenCV、模型和构建目录，不在构建过程中静默下载依赖。
- 第三方二进制依赖放在 Git 忽略的本机目录；仓库只提交源码、构建说明、版本指纹和验证结果。

## 6. C++ CLI 契约

程序名：`ppshitu_rec_benchmark.exe`

必需参数：

```text
--model-dir <ASCII staged model directory>
--image-list <UTF-8 JSON file containing ordered absolute paths>
--report <output JSON path>
```

可选参数：

```text
--threads <positive integer, default 1>
--workers <positive integer, default 1>
--batch-size <positive integer, default 1>
--warmup <non-negative integer, default 50>
--iterations <positive integer, default 200>
--dump-embeddings <optional .f32 binary output>
```

约束：

- `workers > 1` 时，每个 worker 必须拥有独立 `paddle_infer::Predictor`；禁止多个线程并发调用同一 predictor。
- 输入顺序和输出 embedding 顺序必须一致，不能静默跳过不可读图片。
- 任意图片无效、模型文件缺失、输出维度变化或非有限值都以非零退出码失败，并在 JSON 中写入稳定错误码。
- JSON 至少记录 Git 提交、Paddle Inference 版本、OpenCV 版本、模型 SHA-256、CPU 线程/worker、图片 SHA-256、预热次数、迭代次数、特征维度和分阶段延迟。

## 7. 预处理与特征一致性

C++ 路径必须复制当前 Python 路径的实际语义，而不是只照抄旧版示例：

1. 用 Windows 宽字符路径读取文件字节，再通过 `cv::imdecode` 解码，避免中文路径被 `cv::imread(std::string)` 破坏。
2. OpenCV 解码得到 BGR 后显式转换为 RGB。
3. 按当前 `deploy/configs/inference_general.yaml` 使用 224×224、线性插值、`scale=1/255`、现有 mean/std。
4. 将 HWC 转为连续的 NCHW `float32`。
5. 模型输出按样本执行 L2 归一化，维度必须和 Python 输出完全一致。
6. 批量模式不能改变单张结果顺序。

一致性门禁：

- 每张图片的 C++/Python embedding 余弦相似度不得低于 0.9999；
- 每张 embedding 的最大绝对误差不得超过 0.001；
- 特征维度、有限值检查和图片数量必须完全一致；
- 在仓库现有三个数据集及固定五图批次上，最终标签、是否复核和有效几何规则状态必须与 Python 基线完全一致；
- 任一门禁失败时不得进入产品接入。

## 8. 性能实验

### 8.1 对照条件

- 使用同一模型目录、同一图片文件和同一进程优先级。
- Python 与 C++ 都在模型加载后预热 50 次。
- 每种配置至少执行 200 个五图批次；单图至少执行 1000 次。
- 比较配置至少包括：单预测器 1/2/4 线程、五图真实 batch、2×2、4×1。
- 分别记录图片读取、颜色转换和 resize/normalize、模型 `Run()`、L2 归一化、结果序列化及端到端时间。
- 报告 P50、P95、P99、最大值、图/秒和峰值工作集；不把进程启动和模型加载混入热推理。
- C++ 与 Python 的直接性能比较只使用二者共有的“解码 + 预处理 + PP-ShiTu 特征提取 + L2 归一化”边界，禁止拿原生特征提取耗时直接对比包含 TCP、几何、Ridge 和序列化的完整后端耗时。
- 端到端投影使用当前完整 `predict_batch` 的实测 P95，减去其中实测的 Python 特征提取 P95，再加上同批输入的 C++ 特征提取 P95。报告必须同时列出三个原始值和公式，不得只给投影结论。

### 8.2 接入判定

后续产品接入必须同时满足：

1. 第 7 节全部正确性门禁通过；
2. 开发机同口径五图特征提取 P95 相对当前 Python `RecPredictor` 至少下降 15%；
3. 按第 8.1 节公式计算的五图端到端 P95 投影，相对当前完整 `predict_batch` 至少下降 15%；
4. 没有通过提高复核率、关闭几何规则或改变标签语义换取速度；
5. C++ 运行时及依赖可以被离线包审计，许可证与 DLL 来源完整；
6. 后续产品接入完成后，必须重新测量真实完整 `predict_batch`；只有在 i5-8250U 上实测五图 P95 达到 100 ms，才能宣称满足甲方目标，评估阶段的投影不能替代该实测。

若第 2 条不满足，结论应明确写为“C++ 语言迁移收益不足”，后续优先评估 OpenVINO/ONNX Runtime、模型蒸馏或量化，而不是继续扩大 C++ 重写范围。

## 9. 依赖与构建

- 编译器固定为 Visual Studio 2019 x64，匹配当前 Qt 构建链和 Paddle 官方 Windows CPU 预测库。
- Paddle Inference 优先使用与当前 Python Paddle 3.2.2 相同版本的官方 Windows CPU AVX/MKL 包；若官方没有对应包，可用官方 3.x Windows CPU 包进行评估，但报告必须记录版本差异并重新执行全部 embedding 门禁。
- OpenCV 使用 Windows x64 MSVC 开发包；版本必须写入报告。
- i5-8250U 支持 AVX2，CPU 包仍需在启动时检测 AVX，检测失败时输出 `CPU_FEATURE_UNSUPPORTED`，不能崩溃或给出模糊错误。
- 构建脚本检查 `paddle_inference.dll/.lib`、头文件、OpenCV 配置和模型三件套；任何缺失立即失败。
- 构建过程默认完全离线，禁止 CMake `FetchContent` 自动访问网络。

## 10. 测试策略

遵循测试先行：

1. 先写 CLI 参数、JSON 结构、图片顺序和错误码的失败测试。
2. 再实现不加载模型的参数/报告层，使契约测试通过。
3. 先写固定 RGB 小图的 Python 参考预处理向量，再实现 C++ 预处理并比较字节级输入张量。
4. 先写 embedding 对齐测试，再接入 Paddle Inference。
5. 先写单 predictor 与多 predictor 顺序/隔离测试，再实现并行 worker。
6. 最后运行真实模型正确性、当前三个数据集分类回归和性能基准。

不具备本机 C++ 依赖时，纯 Python 契约测试必须仍可运行；依赖真实 C++ 产物的测试必须显式标记为 integration，并在发布验证中强制执行，不能静默算作通过。

## 11. 交付物

本阶段完成时应产生：

- 可复现的 C++ 源码和离线构建脚本；
- 依赖版本与 SHA-256 清单；
- C++/Python embedding 一致性报告；
- 单图和五图各配置性能报告；
- 当前三个数据集的标签/复核一致性报告；
- 一份明确的“接入”或“不接入”结论及证据；
- 推送到 `feature/20260901/ppshitu-cpp-inference` 的完整提交历史。

只有上述交付物和硬门禁均有证据时，本阶段才算完成。

## 12. 回滚与安全性

评估工具与现有产品路径完全隔离；当前 CPU 分支已经推送并保持可独立构建。评估失败不会修改工件库、模板缓存、几何规则或正式离线包。删除本分支即可完整回滚本阶段工作。
