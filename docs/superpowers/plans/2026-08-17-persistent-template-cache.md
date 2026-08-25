# 持久化模板特征缓存实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with checkpoints.

**Goal:** 在不改变预测模型和融合阈值的前提下，缓存并复用未变化工件的模板特征，缩短后端重复启动时间。

**Architecture:** `OrientationClassifier` 生成并校验每个工件的模板签名，负责 `.template_cache.pkl` 的原子读写；`WorkpieceLibrary.recover` 通过可选 loader/saver 回调决定命中缓存或完整重建；`WorkpieceCatalog` 在注册、追加、恢复和启动恢复路径中接入回调。

**Tech Stack:** Python 3.10、pickle、NumPy、pytest、现有 Qt5/Python TCP 协议。

## Global Constraints

- 不修改 PP-ShiTuV2、ALIKED、LightGlue 和现有融合阈值。
- 缓存无效时必须完整重建，不能静默使用过期特征。
- 旧版没有缓存文件的工件库保持兼容。
- 模型和模板恢复完成前，预测和建库继续返回 `MODEL_LOADING`。

### Task 1: 缓存回调边界

**Files:**
- Modify: `src/workpiece_library.py`
- Test: `tests/test_workpiece_library.py`

- [x] 为 `recover` 增加可选 `cache_loader(record)` 和 `cache_saver(record, cache)`，先写测试验证命中时不调用 builder、未命中时重建并保存。
- [x] 运行对应 pytest，确认新测试在实现前失败。
- [x] 实现回调并保持旧调用签名兼容；缓存加载异常回退重建，保存异常只记录警告。
- [x] 运行 `tests/test_workpiece_library.py`，确认通过。

### Task 2: 分类器缓存序列化与签名

**Files:**
- Modify: `src/orientation_classifier.py`
- Test: `tests/test_orientation_classifier.py`

- [x] 先写 round-trip、模板指纹变化失效和损坏文件回退测试。
- [x] 实现版本化签名、CPU 特征校验、pickle 原子写入和无效缓存返回 `None`。
- [x] 运行分类器测试并确认缓存对象包含不等数量模板的全量向量和局部特征。

### Task 3: 生命周期接入

**Files:**
- Modify: `src/workpiece_catalog.py`
- Test: `tests/test_workpiece_catalog.py`

- [x] 先写测试验证恢复使用 classifier 的 loader，注册/追加/恢复后调用 saver。
- [x] 实现统一的可选缓存回调适配；缓存命中后仍按当前活动干扰组重新过滤 raw 特征。
- [x] 运行 catalog、library 和 classifier 测试。

### Task 4: 服务与 Qt 回归验证

**Files:**
- Test: `tests/test_orientation_tcp_service.py`
- Test: `qt_app/tests/test_backendprocessmanager.cpp`
- Modify: `docs/verification/workpiece-orientation-desktop-ui-checklist.md`

- [x] 验证 loading/ready 协议和 Qt 禁止业务请求的行为不变。
- [x] 构建 Qt5，运行现有 Qt 测试和 Python 非集成测试。
- [x] 运行 `git diff --check`，记录旧库首次启动重建与后续缓存命中的差异。
