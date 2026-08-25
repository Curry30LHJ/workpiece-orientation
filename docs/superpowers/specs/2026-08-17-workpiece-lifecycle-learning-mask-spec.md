# 工件生命周期、确认入库与干扰区域传播规格

## Problem Statement

当前系统可以建立工件模板库并执行正反面识别，但没有安全删除工件的能力；操作员确认识别结果正确后，也不能把这张经过人工确认的图片安全地补充到对应正面或反面的模板库。

此外，实际采集图中可能出现反光、阴影、边缘侵入的其他工件等干扰。操作员能够识别这些区域，但逐张模板重复标注成本很高。由于输入图片存在裁剪、旋转和偏移，固定像素坐标的遮罩不能可靠套用到其他图片；若错误屏蔽了工件真实的正反面差异，还会降低识别可靠性。

## Solution

系统为每个工件提供可恢复删除、人工确认后的异步增量入库，以及少量人工框选样本驱动的干扰区域传播功能。这里的“边用边强化”是可解释的模板、掩码和人工反馈积累，不是语义模型训练；系统不会声称能从少量框选中学会“反光”或“阴影”这一抽象类别。

操作员明确选择真实方向后，图片进入持久化异步入库队列。同一工件的短时间连续确认会合并为一次重建；检测继续使用原有稳定模板缓存，只有新模板、特征和清单全部完成后才原子切换。删除会取消该工件未提交的入库任务，并将稳定版本移入回收区。

操作员可在少量模板图上用矩形框标出干扰并归入命名干扰组。系统用现有局部特征对应和几何一致性，把与至少两个可信来源一致的干扰区域投影到其他模板和未来确认图片；高可信候选自动启用，低可信候选进入待复核。第一版只剔除模板侧局部匹配中的干扰关键点，保持查询图、PP-ShiTuV2 全局特征、ALIKED/LightGlue 权重和现有融合阈值不变。

## User Stories

1. As an operator, I want to delete a selected workpiece into a recoverable recycle area, so that an accidental deletion does not destroy accumulated templates.
2. As an operator, I want to restore a deleted workpiece with its original templates and propagated interference annotations, so that it can immediately be used again.
3. As an operator, I want to permanently clear only an explicitly selected recycled workpiece, so that storage cleanup is deliberate.
4. As an operator, I want to see a confirmation before deleting a workpiece, so that I know which library will become unavailable for prediction.
5. As an operator, I want a completed single-image prediction to offer explicit “confirm front and add”, “confirm back and add”, and “do not add” actions, so that a model mistake never becomes a mislabeled template automatically.
6. As an operator, I want the same explicit confirmation actions for each batch-prediction row, so that I can efficiently curate many verified samples without accepting them blindly.
7. As an operator, I want confirmed images to enter a background queue immediately, so that the interface remains responsive while feature extraction runs.
8. As an operator, I want nearby confirmations for the same workpiece to be coalesced, so that a burst of confirmations does not rebuild the same cache repeatedly.
9. As an operator, I want prediction to continue during background ingestion, so that production inspection does not pause while templates are improving.
10. As an operator, I want a background ingestion failure to leave the previous stable workpiece library intact, so that a bad image or model error cannot break inspection.
11. As an operator, I want to view queued, building, completed, failed, cancelled, and review-required ingestion jobs, so that I can understand what happened to every confirmed image.
12. As an operator, I want to retry or cancel an eligible failed or queued job, so that I can resolve transient errors without redoing confirmation work.
13. As an operator, I want a newly confirmed image whose interference mask cannot be inferred safely to remain pending review instead of entering the active library, so that interference is not reintroduced silently.
14. As an operator, I want to draw one or more rectangular interference regions on selected template images, so that I can identify glare, shadow, or invading workpiece evidence I do not want used in local matching.
15. As an operator, I want to name and manage multiple interference groups, so that distinct recurring disturbances such as an edge reflection and an invading workpiece are not conflated.
16. As an operator, I want an interference group to require at least two trusted marked examples before automatic propagation begins, so that a one-off artifact is not incorrectly generalized.
17. As an operator, I want high-confidence propagated masks to become active automatically and be visibly marked as automatic, so that I avoid drawing the same interference repeatedly.
18. As an operator, I want low-confidence propagated masks to remain inactive and visible for review, so that uncertain correspondence never changes recognition silently.
19. As an operator, I want accepting or correcting an automatic mask to add trusted evidence and rejecting it to prevent the same unchanged proposal for that target, so that feedback has a precise effect without pretending to retrain a semantic model.
20. As an operator, I want trusted interference groups to be eligible for both orientations and future confirmed templates of the same workpiece when correspondence is reliable, so that recurring interference is handled consistently.
21. As an operator, I want the system to warn and refuse unsafe masks that leave too little usable local evidence, so that true front/back characteristics are not accidentally excluded.
22. As an existing deployment owner, I want old template libraries without ingestion metadata or interference annotations to load normally, so that this upgrade does not invalidate prior work.
23. As an existing TCP client owner, I want existing commands and responses to remain valid, so that deploying the new desktop application does not break older integrations.
24. As an operator, I want confirmation actions to remain bound to the workpiece, image digest, and prediction result that produced them, so that switching workpieces or running another prediction cannot add an image to the wrong library.
25. As an operator, I want repeated submission after a connection timeout to return the original ingestion or lifecycle result, so that network retry cannot create duplicate jobs or delete twice.
26. As an operator, I want an interrupted background build to resume safely after backend restart, so that durable confirmed images are not lost and a partial revision never becomes active.
27. As an operator, I want restoration to stop with a clear conflict if another active workpiece now uses the same name, so that restoring an old library never overwrites a new one.
28. As an operator, I want exact duplicate images rejected before they enter the queue and near-duplicate or imbalanced template counts shown as warnings, so that the library does not grow silently while retaining operator control.
29. As an operator, I want automatic mask diagnostics to explain which trusted examples agreed, why a candidate was accepted or held for review, and how much local evidence remains, so that automatic activation is auditable.
30. As an operator, I want unsaved annotation edits protected when changing image, group, workpiece, or closing the editor, so that manual regions are not discarded accidentally.
31. As an operator, I want completed and cancelled job payload images removed while their small status records remain until I clear history, so that diagnostics remain available without unbounded image storage.
32. As an operator, I want a new or edited interference group to remain a draft until all affected templates are resolved, so that recognition never uses a partially propagated group.
33. As an operator, I want to mark an interference group as absent on a particular template, so that a clean image is not blocked merely because no propagated region was found.
34. As an operator, I want replacing or restoring a workpiece to invalidate stale confirmations and stale background revisions, so that old inspection context cannot mutate a different library generation.

## Implementation Decisions

### Deep Modules and Seams

- The **Workpiece Catalog module** owns active/recycled lifecycle, immutable revisions, manifest migration, atomic persistent/runtime publication, restore conflict detection, and permanent removal. Its **interface** exposes coherent snapshots, lifecycle transitions, and compare-and-commit; callers never coordinate directory moves, backup recovery, classifier cache replacement, or cache removal themselves. The existing persistent library and classifier cache become implementation details behind this seam. Tests use a real temporary local filesystem plus a fake cache publisher, rather than exposing filesystem mechanics in the interface.
- The **Template Evolution module** owns explicit confirmation intake, durable staging, idempotency, three-second coalescing, job state transitions, mask propagation, candidate cache construction, review staging, retry/cancel, and atomic cache publication. Its small **interface** accepts a confirmation, lists job snapshots, and applies one valid action to a job. Queue mechanics, workers, clocks, and propagation policy are implementation details behind this seam.
- Geometric mask projection is an internal seam of the Template Evolution module. Production uses the existing ALIKED/LightGlue matcher as an **adapter**; deterministic fake correspondences are the second adapter for tests. Raw matcher structures are not exposed through the module interface.
- TCP and Qt are adapters over these two domain interfaces. They translate commands and render snapshots but do not implement lifecycle rules, job transitions, mask confidence, or commit ordering. MainWindow remains an orchestrator; job management and image annotation live in focused dialogs/widgets rather than expanding MainWindow with their internal state.

### Lifecycle and Revision Consistency

- Workpiece lifecycle has three persistent states: active, recycled, and permanently removed. Active workpieces are the only ones listed for prediction. Recycled workpieces retain their identifier, last committed revision, templates, annotations, and metadata.
- Every workpiece has a monotonically increasing revision covering both content and lifecycle transitions. Every background job captures a base revision or an explicit predecessor job. Commit succeeds only when the workpiece is active and its expected lineage resolves to the current revision. Deletion, restore, replacement registration, and successful ingestion change the revision; a worker with an old token or failed predecessor becomes stale and may never publish files or cache.
- Deletion is a confirmed, recoverable transition. It records cancellation intent for all unfinished jobs, prevents new commits through the revision guard, removes the active prediction cache, and atomically moves the last committed revision into the recycle area. A worker already extracting features may finish disposable computation, but its final compare-and-commit must fail.
- Restore preserves the original workpiece identifier and committed content. It fails without modifying either library when an active workpiece has the same identifier or case-insensitive display name. Permanent removal is available only for recycled workpieces and also removes their pending review payloads and nonessential job history after confirmation.
- Startup recovery resolves incomplete lifecycle transactions before exposing the workpiece list. It never guesses between two complete conflicting revisions; the ambiguous workpiece remains unavailable and produces a diagnostic requiring operator action.

### Confirmation and Durable Jobs

- Human confirmation is always an explicit operator label. A predicted result is never accepted automatically; single prediction and every batch row expose front, back, and no-add choices.
- Each confirmable result is an immutable result context containing workpiece identifier and revision, source image path, decoded-content digest, prediction request identifier, and displayed model result. Changing selected workpiece, changing the source image, or replacing the displayed result invalidates its confirmation controls. Durable staging is created only when confirmation is submitted successfully.
- Every mutating TCP request carries a caller-generated operation identifier. The backend persists the operation result and returns the same result when that operation identifier is retried, including after reconnect or restart. Request identifiers still correlate one transport exchange and do not replace mutation idempotency.
- Confirmed input is decoded, validated, hashed, checked against active and unfinished templates, and copied into durable job staging before enqueue succeeds. Exact decoded-content duplicates are rejected. Near duplicates and material front/back count imbalance produce warnings but do not impose a hard limit.
- Ingestion is coalesced per workpiece across a three-second quiet period while retaining the confirmed orientation of every image. A job already building is immutable. Later confirmations create a successor that explicitly depends on the running job: if the predecessor completes, the successor uses its revision; if the predecessor fails or is cancelled without publication, the successor may use the unchanged base revision; any unrelated revision change makes the successor stale and requires operator retry/reconfirmation.
- Externally visible job states are `queued`, `building`, `needs_review`, `completed`, `failed`, and `cancelled`. Valid transitions are enforced by the Template Evolution module. Retry moves an eligible failed job to queued; review resolution moves an eligible needs-review job to queued or cancelled; cancellation of building work records intent and ends as cancelled only after publication is impossible.
- On backend restart, a persisted building job returns to queued with the same staged inputs and operation identity. Completed and cancelled payload images are removed while compact job records remain until the operator clears history. Failed jobs retain only data required for retry; needs-review jobs retain their staged image and candidate annotations.
- A background worker builds candidate templates, metadata, masks, raw local features, filtered prediction features, and the full candidate cache outside the active snapshot. It validates against the complete base revision and atomically publishes the new workpiece revision and cache only after every artifact succeeds. Prediction reads one immutable cache snapshot throughout; unsuccessful work leaves the old revision untouched.

### Interference Groups and Mask Propagation

- Workpiece metadata is versioned. A version-2 manifest records immutable revision, templates with stable copied-image identity, decoded-content digest, native dimensions, confirmed orientation, ignored regions, annotation provenance, and interference groups. A version-1 manifest remains readable and is upgraded only by a successful transactional mutation, never merely by recovery.
- Manual rectangular selections are converted to image-native quadrilateral regions with source dimensions. Display-to-image mapping accounts for aspect-fit padding, zoom, and scroll; saving and reopening must reproduce the same native region. Unsaved edits prompt before image, group, workpiece, or window changes.
- An interference group belongs to one workpiece and has a user-visible name plus trusted manual/accepted/corrected annotations. At least two trusted source annotations are required before propagation. Rejected candidates are recorded for that target and group so the same proposal is not reactivated unchanged; rejection is not represented as semantic model training.
- Interference-group editing uses a durable draft annotation revision. Creating, changing, disabling, or deleting a group does not partially mutate the active prediction cache. The draft is propagated and reviewed first; one complete candidate cache is built and atomically published only after every affected template is resolved and safety validation passes.
- Propagation extracts unfiltered local features and estimates source-to-target geometry independently from trusted examples. An automatic candidate is high confidence only when at least two independent trusted sources produce geometrically valid, mutually consistent projected regions and the central safety policy accepts projected area and remaining local evidence. Cross-orientation propagation is allowed only when the same target independently passes these checks.
- The propagation safety policy and its diagnostic reasons are localized inside the Template Evolution module and are distinct from recognition fusion thresholds. Every candidate stores source evidence, geometric diagnostics, consensus result, active/pending state, and rejection reason where applicable so automatic activation is auditable.
- High-confidence candidate masks are active and visibly labelled automatic. Low-confidence, conflicting, oversized, or evidence-starving candidates remain inactive in needs-review. Accepted or corrected candidates become trusted group evidence. A nonrepeatable glare or shadow that cannot achieve geometric consensus will remain manual or pending; the system does not claim semantic recognition of that artifact.
- Every enabled interference group has one target status per template: active mask, operator-confirmed absent, or unresolved. Absence is never inferred merely because propagation found no match. A group revision and a newly confirmed image may enter the active library only after every enabled group is active or explicitly confirmed absent for every affected target; otherwise the job remains needs-review and the active revision is unchanged.
- Raw template local features are used during propagation; active masks are then applied to produce the filtered template features stored in the prediction cache. Ignore regions remove template-side ALIKED keypoints and aligned descriptors/scores before LightGlue scoring. Query-side features, PP-ShiTuV2 embeddings, model weights, model input size, decision policy, and existing fusion thresholds remain unchanged.

### Protocol and Desktop Behavior

- 干扰标注管理与递推审计界面采用独立的细化设计，详见 `2026-08-17-interference-annotation-management-design.md`。该设计补充类型启停/删除、模板原图叠加、逐模板递推来源与诊断、草稿/生效版本区分和基于修订号的并发保护；这些约束优先于仅提供单次矩形输入的简化界面。
- Protocol version 1 gains additive commands `recycle_workpiece`, `list_recycled_workpieces`, `restore_workpiece`, `purge_workpiece`, `submit_confirmation`, `list_evolution_jobs`, `evolution_job_action`, `get_workpiece_annotations`, `save_workpiece_annotations`, `set_workpiece_annotation_group_enabled`, and `delete_workpiece_annotation_group`. `evolution_job_action` accepts only state-valid actions such as cancel, retry, resolve-review, or clear-history. Existing hello, list, register, predict, progress, loading, and shutdown behavior remains valid. Older clients may ignore added list fields.
- Mutation responses return operation identifier, affected workpiece identifier and revision, and a domain snapshot. Job list responses return stable job identifiers, state, progress, warnings, review reason, and retry/cancel eligibility. Qt polls job snapshots only while the client is Ready and never creates a second pending TCP request.
- Idempotency records live in a compact ledger independent of workpiece and job payloads, survive recycle, purge, reconnect, restart, and job-history clearing, and are not automatically expired by this feature. Reusing an operation identifier with a different payload is rejected as an invalid request rather than replaying an unrelated result.
- Stable new errors include `WORKPIECE_RECYCLED`, `WORKPIECE_NOT_FOUND`, `RESTORE_CONFLICT`, `STALE_WORKPIECE_REVISION`, `DUPLICATE_TEMPLATE`, `JOB_NOT_FOUND`, `JOB_NOT_ACTIONABLE`, `INVALID_MASK`, and `MASK_PROPAGATION_UNSAFE`. Errors use stable codes and user-facing Chinese messages; internal paths and tracebacks remain in backend logs.
- The desktop adds explicit confirmation controls bound to immutable result context; a recycle confirmation and recycle-management dialog; a job dialog with status, retry, cancel, review, and clear-history actions; and a template annotation editor with group management, native-coordinate rectangle editing, provenance indicators, diagnostics, and unsaved-change protection.

## Testing Decisions

- Tests verify externally observable behavior through the Workpiece Catalog and Template Evolution interfaces: coherent lifecycle/cache snapshots, revision publication, job snapshots, review outcomes, and prediction visibility. They do not assert private locks, thread order, internal helper calls, or exact staging names.
- Workpiece Catalog interface tests cover recycle/restore/permanent removal, persistent/runtime publication coherence, case-insensitive restore conflicts, startup recovery, legacy manifest reading, version-2 mutation, expected-lineage rejection, and stale-worker commit prevention. A temporary local filesystem provides real rename and crash-recovery semantics; no filesystem port is added solely for tests.
- Template Evolution interface tests inject a fake clock, deterministic executor, fake feature builder, and fake geometric matcher. They cover three-second coalescing without real waits, operation-id idempotency, exact duplicate rejection, restart recovery, immutable running jobs, successor jobs, every valid/invalid state transition, cancellation during build, needs-review resolution, atomic cache publication, and old-cache preservation after failure.
- Mask behavior is principally tested through the Template Evolution interface. Deterministic correspondence fixtures cover two-source agreement, cross-orientation valid geometry, source disagreement, rejected-candidate replay prevention, area/evidence safety rejection, native-coordinate round trips, and filtered-keypoint output. A small internal pure-geometry test is permitted only for transform mathematics that cannot be diagnosed clearly through the higher seam.
- TCP adapter tests verify each additive command, mutation idempotency, stable error mapping, revision fields, job snapshots, polling while Ready, and compatibility of existing commands. The existing JSON-lines dispatcher/server tests are the prior art; domain behavior is not duplicated exhaustively at this layer.
- Desktop adapter tests use QTest with the existing fake TCP server and fake process launcher. They cover result-context invalidation, deletion confirmation, recycle conflict presentation, front/back confirmation from single and batch outputs, job controls, annotation coordinate mapping, unsaved-change prompts, automatic/pending provenance, diagnostics, and disabled states while the client has a pending request.
- Compatibility tests load unchanged version-1 workpiece manifests and verify prediction remains available. The first successful mutation writes a recoverable version-2 revision. Legacy clients continue using hello, list, register, predict, and shutdown without new fields.
- Real-model verification runs only when configured PP-ShiTuV2 and ALIKED/LightGlue assets are available. It records mask propagation diagnostics and verifies active masks change only local template evidence while global embeddings, model configuration, and fusion thresholds remain unchanged. Before high-confidence automatic activation is released, reviewed project images must be used to calibrate the propagation safety policy and archive false-activation examples; synthetic tests alone are insufficient.

## Out of Scope

- Training or fine-tuning PP-ShiTuV2, ALIKED, LightGlue, a detector, or a semantic segmentation model.
- Treating a fixed display-coordinate rectangle as a universal query-image mask.
- Claiming that local masking fully removes query-side or PP-ShiTuV2 global-feature sensitivity to reflection, shadow, or occlusion.
- Automatic admission of a prediction based only on the model's predicted label.
- Automatic timed cleanup of recycled workpieces.
- Cross-workpiece interference learning; groups are isolated to one workpiece.
- Changing existing global/local fusion thresholds or deciding labels with a new confidence policy.
- Semantic recognition of “reflection”, “shadow”, or “invading workpiece” from a few boxes; propagation requires repeatable local geometry.
- Query-side masking, image alignment for PP-ShiTuV2, or any claim of global-feature invariance.
- Multiple simultaneous Qt clients or remote multi-user conflict resolution; the backend remains a loopback single-client application with one background evolution worker.
- Automatic expiry or administrative compaction of the mutation-idempotency ledger.

## Further Notes

- The practical first release deliberately uses correspondence-based propagation rather than few-shot detector training. It is compatible with the current fixed-camera but variably cropped/rotated input only when repeatable local geometry can be established, and it produces inspectable per-template results.
- If measurements later show that PP-ShiTuV2 global evidence remains the dominant source of error after local masking, a separate future design is required for object-coordinate alignment and query-side global masking. That work is intentionally not hidden inside this feature.
- Delivery is sliced without weakening invariants: first Workpiece Catalog lifecycle and recycle UI; then durable Template Evolution jobs and explicit confirmation; finally annotation editing, propagation, review, and automatic activation. Each slice is independently testable and keeps existing protocol behavior usable.
- The highest domain test seams are the Workpiece Catalog and Template Evolution interfaces. TCP and Qt remain adapters tested for translation and interaction, which concentrates concurrency, persistence, cache publication, and mask rules in the deep modules rather than repeating them across callers.
