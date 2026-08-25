# 工件生命周期、确认入库与干扰区域实现验证

日期：2026-08-17

## 已实现

- `WorkpieceCatalog` 收口 active/recycled 生命周期、revision 和 classifier cache 发布；支持回收、恢复、永久清除，恢复保留原工件 ID，并检查大小写不敏感的名称冲突。
- `TemplateEvolution` 使用 `jobs.json` 和 staging 图片实现显式 front/back 确认、精确内容去重、3 秒合并、后台单 worker、重启时 building→queued、失败保留旧缓存、任务取消/重试/复核状态。
- `interference_masks.py` 提供原生坐标矩形、两可信来源几何一致性、面积安全门和模板侧 ALIKED 特征过滤；全局 PP-ShiTu 特征、查询特征和融合阈值未改动。
- TCP 新增回收/恢复/清除、确认入库、任务列表/动作、标注读取/保存命令；旧 hello/list/register/predict/shutdown 路径保持兼容。
- Qt 新增删除二次确认、单张和批量“确认正面/确认反面/不入库”、任务状态轮询、模板干扰框选编辑器；已有模板数量和建库进度界面保留。
- Qt 干扰标注已升级为独立管理器：可查看每个模板的人工/自动/待复核区域、递推来源与诊断，支持新增/重命名、编辑多矩形、类型停用/启用/删除及复核动作；主窗口通过快照和 `base_revision` 串行编排后端请求。
- 标注文档增加 draft/active 双修订和旧 manifest 兼容读取；启动恢复通过 `WorkpieceCatalog.recover()` 重建已生效掩码，待复核草稿不会混入预测缓存。

## 验证结果

### Python

- `python -m pytest tests/test_workpiece_catalog.py tests/test_template_evolution.py tests/test_orientation_tcp_service.py tests/test_interference_masks.py -q -p no:cacheprovider`：31 passed。
- 排除当前环境未安装 `lightglue` 的模型依赖测试后，显式文件集合：73 passed。
- `python -m py_compile src/workpiece_library.py src/workpiece_catalog.py src/template_evolution.py src/interference_masks.py src/orientation_classifier.py src/orientation_tcp_service.py`：通过。
- `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q`：94 passed, 3 skipped（157.43 秒）；跳过项为当前环境缺少可选真实模型依赖的测试。
- `git diff --check`：通过（仅有仓库既有 CRLF 提示）。

### Qt 5.14.2 / MSVC 2019

- `scripts/build_qt5.ps1`：release 构建成功，产物为 `qt_app/build-release/release/workpiece_orientation.exe`。
- `test_mainwindow.exe -platform offscreen`：通过，包含模板数量、批量检测和生命周期/确认/标注控件测试。
- 新增 `test_annotationmanager.exe -platform offscreen`：6 passed；验证原图坐标映射、多矩形选择/删除、快照状态/诊断和删除/复核信号。
- `test_mainwindow.exe -platform offscreen`：12 passed；主窗口回归覆盖通过快照打开管理器所需的原有状态与请求串行约束。
- 既有 AppConfig、BackendClient、BackendProcessManager 和 MainWindow 测试可执行文件：通过。

## 依赖限制与性能说明

- 当前 Python 环境没有 `lightglue` 包，因此 ALIKED/LightGlue 真实模型测试未执行；生产环境已有模型依赖时应再运行真实图片回归。
- 确认入库采用完整候选缓存重建，建库时间随正反面模板总数近似线性增加；3 秒合并可避免连续确认重复重建。预测在后台重建期间继续使用旧稳定缓存，提交成功后才切换。
- 干扰传播仅在现有 ALIKED/LightGlue 局部对应和几何安全门通过时自动激活；低置信度或未解析目标进入 needs_review，不会静默当作“无干扰”。
- 标注修改不改变 PP-ShiTuV2、ALIKED、LightGlue、512 关键点上限或现有融合阈值；掩码只作用于模板侧局部特征。
