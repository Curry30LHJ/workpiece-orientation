# 几何标定干扰屏蔽验收记录

日期：2026-08-17  
分支：`feature/20260815/workpiece-orientation-desktop-ui`

## 变更范围

本轮实现把旧的模板像素位置递推替换为对象相对几何规则。规则支持圆、椭圆和旋转矩形，分别可以忽略内部或外部区域；每张模板和查询图都会重新拟合边界。PP-ShiTuV2、ALIKED、LightGlue 的权重、特征维度和现有融合阈值未修改。低置信度拟合会回退到无遮罩基线并标记 `needs_review`。

关键提交：

- `4a57cd7` 对象相对几何标定；`f4f9d07` 全局/局部推理遮罩接入；
- `ccd160c` 草稿与版本化档案；`a3a8aa9` 后台验证、发布与回退；
- `64037fc` 恢复和确认入库守卫；`a2eba9d` TCP profile 命令；
- `16b41e4` Qt 形状画布；`bab5a75` 缓存留一验证；
- `f19409a` Qt 规则管理、验证轮询和显式发布；`77cb0f2` 模板预览、基准标定和像素到对象坐标换算修正。

## 环境

- Windows 10/11，Python：`E:\python\anaconda3\envs\shitu\python.exe`
- Qt：5.14.2 MSVC 2017 64-bit，Visual Studio 2019 x64 工具链
- 真实模型目录：`third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer`
- Qt Release 输出：`qt_app\build-release\release\workpiece_orientation.exe`

## Python 测试

命令：

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q --basetemp=pytest-temp-full-20260817
```

结果：`139 passed, 3 skipped in 76.67s`。

几何相关目标回归（几何标定、分类器、profile、catalog、模板演进和 TCP）结果：`88 passed in 118.43s`。新增缓存留一验证单测结果：`1 passed, 16 deselected in 19.34s`。

3 个跳过项属于已有的真实服务集成测试，不是本次几何规则失败；全量测试没有失败。

## Qt 构建和测试

构建命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_qt5.ps1
```

结果：构建成功，生成 `qt_app\build-release\release\workpiece_orientation.exe`。

在 `qt_app\tests` 目录使用 Qt 5.14.2 + MSVC x64 重建并运行以下程序，均以 `-platform offscreen` 返回退出码 0：

- `test_geometryrulecanvas.exe`
- `test_geometrymaskmanager.exe`
- `test_mainwindow.exe`
- `test_appconfig.exe`
- `test_backendclient.exe`
- `test_backendprocessmanager.exe`
- `test_annotationmanager.exe`（旧位置标注归档查看兼容性）

QtTest 在当前无窗口会话不输出详细计数，但构建和进程退出码均为 0；管理器测试覆盖发布前置条件、告警覆盖原因和规则增删，主窗口测试覆盖新按钮文案。

几何管理器会从 profile 快照加载当前方向的代表模板，画布绘制结果先转换为原图坐标，再依据 anchor 转成对象相对的 `geometry`；“将画布设为基准边界”用于更新方向 anchor，旋转矩形角度由数值控件写入。

## 后端 smoke

命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\smoke_orientation_service.ps1 -StartupTimeoutSeconds 600
```

结果：真实 PP-ShiTuV2/ALIKED/LightGlue 服务完成模型加载，`hello`、`list_workpieces`、`shutdown` 均通过。首次启动因模板缓存签名升级而重建已有 28+28 工件缓存，启动等待明显长于纯协议测试；后续命中有效缓存后会缩短。脚本新增可选参数 `-GeometryWorkpieceId` 和 `-GeometryPredictImage`，用于在不修改默认库的前提下执行 profile 草稿、验证、发布、预测和回退全链路。

当前真实库 `runtime_library\2423857f5be7432bb8750d98d73036af` 仍只有旧 `interference_groups`，没有已发布几何 profile。为避免验收脚本改写用户工件库，本次没有对该库自动发布空的几何规则；需人工准备几何草稿后再运行可选生命周期参数。

## 性能与准确率记录

本次自动化验收没有伪造 35+35 工件的准确率。现有可复现结论如下：

- 模板候选构建阶段每张模板只做一次遮罩后的全局/局部特征提取；留一验证只消费缓存，不再次提取模型特征。
- 预测阶段每个方向各拟合一次查询几何边界；拟合成功会增加一次遮罩填充和一次方向特征提取，拟合失败直接使用原始基线并提示复核。
- 后台验证通过 job 返回和 Qt 轮询，不占用 Qt 主线程；预测与发布使用同一运行时锁，发布失败不会交换半成品缓存。
- 真实 35+35 工件的建库耗时、遮罩拟合失败数、准确率、p50/p95 预测延迟和缓存文件大小，必须在现场选定几何规则后记录；当前自动化环境没有经过确认的 35+35 几何标注，因此不把旧位置递推数据冒充为新方案结果。

## 复核重点

1. 在 Qt 中对正、反面分别配置 anchor 和规则，保存草稿后点击“验证草稿”。
2. 等待报告显示 `completed`，检查每个模板的 fit 状态、忽略比例、关键点保留率和留一回归；有告警时必须填写覆盖原因。
3. 点击“发布规则”后，在预测结果中确认 `geometry_mask.status=active` 和 profile revision；低置信度样本应显示“遮罩未启用（需复核）”。
4. 如需验证协议全链路，可执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\smoke_orientation_service.ps1 `
  -GeometryWorkpieceId '<工件ID>' `
  -GeometryPredictImage '<待测图片路径>'
```
