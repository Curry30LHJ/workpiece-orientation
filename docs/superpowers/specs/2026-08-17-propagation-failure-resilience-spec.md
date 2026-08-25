# 干扰区域递推失败隔离与可诊断性规格

## Problem Statement

操作员在同一干扰类型中完成第二张可信模板的区域标注后，系统开始向其他模板递推区域。当前任意一对源模板和目标模板之间的局部特征投影异常会穿透递推流程，并被服务端折叠为英文 `Internal server error`。操作员无法知道是哪张模板失败、是否保留了已画的种子区域、其余模板是否仍在处理，以及应当重试、修正还是重启模型。

截图中进度到达 `2 / 56` 后失败，说明两个已标注种子已被跳过，首个未标注模板的投影、目标读取或区域融合发生了未捕获异常。系统必须把单模板的可恢复问题与本地特征模型不可用的问题区分开来，且不能以部分递推结果覆盖正在用于识别的稳定掩码。

## Solution

将干扰区域传播设计为“每个目标模板独立隔离、整组草稿原子提交”的操作。局部匹配不足、目标图片不可读、几何投影不可求，或单对模板的可恢复处理异常，只使该目标模板进入待复核，并记录结构化诊断；其他模板继续递推。GPU 内存不足、CUDA/设备失效或局部特征模型不可用属于致命错误，操作会停止、返回稳定的中文 `MODEL_ERROR`，且不提交新的草稿或活动掩码。

服务为每次传播记录关联操作标识、工件、干扰类型、源模板、目标模板、异常类别与堆栈到本地轮转诊断日志。Qt 在递推期间显示进度；成功但有失败模板时展示待复核数量及逐模板原因；致命失败时恢复控件并显示可执行的中文提示。已确认的人工种子、旧版工件库和当前稳定识别缓存保持不变。

## User Stories

1. As an operator, I want one failed template projection not to cancel the whole interference propagation, so that a single difficult image does not discard my work.
2. As an operator, I want the two manually marked seed templates to remain saved after automatic propagation encounters a difficult target, so that I can continue from the work already completed.
3. As an operator, I want every unpropagated target to appear as pending review with a concise cause, so that I know exactly which image needs attention.
4. As an operator, I want automatic propagation to continue for unaffected templates, so that I do not have to draw every region manually.
5. As an operator, I want progress to continue through all target templates even when some targets fail recoverably, so that the displayed progress accurately represents the operation.
6. As an operator, I want a clear Chinese model error when GPU memory, CUDA, or the local feature model fails, so that I do not mistake an infrastructure problem for a bad annotation.
7. As an operator, I want controls re-enabled after an unsuccessful propagation request, so that I can retry, correct a seed, or restart the backend.
8. As an operator, I want the current stable mask revision to remain in use until a complete, safe propagation result is published, so that inspection cannot silently use a partial mask.
9. As an operator, I want the draft to retain valid manual annotations even if some automatic proposals are unresolved, so that later review can complete the group without repeating seed annotation.
10. As an operator, I want source-template and target-template diagnostics to be visible without exposing Python stack traces in the production UI, so that I can make a recovery decision safely.
11. As a maintenance engineer, I want a local diagnostic log correlated by operation and workpiece, so that I can find the exact failing pair after a field failure.
12. As a maintenance engineer, I want fatal model failures to use `MODEL_ERROR` rather than `INTERNAL_ERROR`, so that monitoring and Qt can distinguish device recovery from ordinary template review.
13. As an existing user, I want existing workpiece libraries and their manually saved annotations to load unchanged, so that this resilience update does not require rebuilding a library.
14. As an existing TCP client, I want version-1 commands and response shapes to remain compatible, so that older integrations continue to work.
15. As a validation engineer, I want deterministic tests for recoverable and fatal projection failures, so that a later change cannot reintroduce a generic internal error.

## Implementation Decisions

- Template Evolution remains the owner of propagation policy. It receives projection outcomes, decides whether a target is active or awaiting review, updates draft groups, and performs the only atomic commit. TCP and Qt do not implement propagation policy.
- A projection pair has three meaningful outcomes: a usable projected rectangle, no valid correspondence, or a recoverable failure. A recoverable failure has a stable reason code and sanitized summary; it never supplies a rectangle or counts as corroborating evidence.
- Expected image and geometry conditions, including unreadable target image, insufficient matches, invalid keypoint indexing, invalid correspondence geometry, and OpenCV geometry failure, are handled at pair or target granularity. Their target becomes unresolved while other targets continue.
- GPU out-of-memory, CUDA/device loss, and inability to invoke the ALIKED/LightGlue model are fatal local-feature failures. The operation aborts before commit and returns a localized `MODEL_ERROR`.
- Unknown pair-level exceptions are logged with stack trace and initially isolated to that pair. A bounded error budget prevents a broken local-feature pipeline from being silently converted into an entire library of unresolved templates; exceeding the budget produces localized `MODEL_ERROR` with no persistence change.
- Target diagnostics include target identity, stable reason code, source attempt count, successful projection count, recoverable projection-failure count, and failed source identities. UI responses never include raw paths outside the workpiece or stack traces.
- Progress emits once for every terminal target outcome, including skipped seeds and recoverable failures. A fatal error stops progress and returns an error response rather than a false completion event.
- A successful operation with unresolved targets commits only the draft annotation document and does not publish a new active interference group. The existing active annotation revision and classifier cache remain active until the group is complete and safe.
- The service writes structured rotating diagnostics under the configured local library area. Log entries carry operation, workpiece, group, source template, target template, exception class, and stack trace; logs are never sent across TCP.
- Qt keeps transient progress and persistent operation failure status separate. An error clears busy state but remains visible until a successful fresh annotation snapshot; target diagnostics are localized for display.
- The change is additive: no model weights, PP-ShiTuV2 preprocessing, ALIKED/LightGlue configuration, fusion threshold, cache format, manifest requirement, or legacy protocol command changes.

## Testing Decisions

- The highest behavioral seam is the Template Evolution public save operation with a deterministic classifier adapter. It covers two trusted seeds, automatic target propagation, draft persistence, and progress without production model weights.
- A recoverable-failure test asserts operation completion, terminal progress for every target, a failed target with diagnostics, automatic candidates for unaffected targets, persistent manual seeds, and unchanged active revision.
- A fatal-failure test asserts a stable domain error, no draft or active revision mutation, and no false completion progress.
- TCP tests use the existing JSON-lines test server to verify stable error translation, progress ordering, request identity, and service availability after either outcome.
- Qt tests use QTest dialog and fake-TCP seams to verify progress visibility, restored controls, localized failure text, and target diagnostic rendering without a GPU.
- Existing propagation, TCP progress, annotation-manager, and main-window tests are prior art. Focused tests run first, followed by the Python suite and Qt 5.14.2 offscreen tests.

## Out of Scope

- Changing PP-ShiTuV2, ALIKED, LightGlue weights, device settings, model input size, keypoint limit, or recognition fusion thresholds.
- Retraining, fine-tuning, semantic segmentation, detector training, or learning a generic visual concept such as reflection.
- Retrying failed targets automatically without an operator request.
- Publishing a partially resolved interference group as the active recognition mask.
- Requiring a workpiece rebuild, migrating old annotations merely for diagnostics, remote log upload, cloud telemetry, multiple backend clients, or a new propagation job queue.

## Further Notes

- The observed `2 / 56` failure is consistent with the first unmarked target invoking local projection after the two marked seeds. A deterministic harness reproduced the essential failure: an exception from one projection pair currently escapes propagation and becomes `Internal server error` at the TCP boundary.
- The exact production exception must be recorded in the proposed local diagnostic log rather than guessed from a screenshot. The resilience behavior does not depend on guessing it.
- The selected test seam is Template Evolution, with TCP and Qt adapter tests. This preserves one authoritative propagation policy and avoids duplicating error classification in the UI.
