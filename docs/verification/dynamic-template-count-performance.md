# 动态模板数量验证与性能影响

## 验证范围

本次变更使用可注入的特征构建器覆盖了 1+1、5+10、10+15、5+12、31+1 和 101+1 模板组合。测试确认所有图片都被复制、写入 manifest、加入缓存，并且旧版缺少 `template_counts` 的 5+5 工件库可以恢复。

运行命令（Windows 本机需要使用可写的 pytest 临时目录；本次运行通过临时测试夹具避开系统临时目录 ACL，夹具未提交）：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_orientation_classifier.py tests/test_workpiece_library.py tests/test_orientation_tcp_service.py -q
```

结果：`51 passed`。Qt 5.14.2 offscreen 测试结果为：AppConfig `5/5`、BackendClient `11/11`、BackendProcessManager `8/8`、MainWindow `11/11`。

## 建库耗时影响

建库特征提取的工作量与 `N = 正面模板数 + 反面模板数` 线性增长：每张模板恰好执行一次全局特征提取和一次局部特征提取。动态数量没有额外的重复提取或隐式截断；进度事件中的 `total` 就是 `N`。因此，与原来的 5+5 相比，5+10 约为 1.5 倍特征提取量，10+15 约为 2.5 倍，31+1 约为 3.2 倍。磁盘复制和图片校验同样按 `N` 线性增长。

本次测试使用 fake builder 验证了调用次数和完整性，没有加载生产 PP-ShiTuV2/ALIKED/LightGlue 权重，因此不把测试机耗时冒充真实模型基准。部署前应在目标工控机上记录每个阶段的耗时；Qt 已显示校验、复制、特征提取、提交阶段和总耗时。

## 预测耗时与显存

局部匹配仍对两面全部模板进行评分，所以单次预测的局部匹配时间也随 `N` 线性增长，融合算法和阈值没有改变。实现改为一次只把一个局部模板候选移动到计算设备并评分，避免把全部候选同时复制到 GPU；因此模板数量增加时峰值候选显存保持近似常数，但总计算时间会增加。全局特征仍对所有模板取最大相似度，复杂度同样为 `O(N)`。
