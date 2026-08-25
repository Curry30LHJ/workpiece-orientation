# 干扰标注管理与递推审计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有工件正反识别桌面程序补齐可查看、可编辑、可启停/删除、可复核并能审计递推来源的干扰标注管理闭环，同时保证标注清单与运行时预测缓存原子一致。

**Architecture:** `WorkpieceLibrary` 负责版本化标注文档的持久化与旧清单兼容；`WorkpieceCatalog` 生成完整只读快照并原子发布有效掩码缓存；`TemplateEvolution` 负责递推诊断、待复核动作和类型生命周期；TCP 只翻译命令。Qt 使用独立 `AnnotationManagerDialog`，复用抽出的 `AnnotationCanvas` 显示/编辑原图坐标区域，`MainWindow` 只串行编排后端请求。

**Tech Stack:** Python 3.10、OpenCV、NumPy、pytest；Qt 5.14.2 Widgets/Network/Test、qmake、MSVC2019（`msvc2017_64` Qt 包）、jom；现有 JSON-lines TCP protocol v1。

## Global Constraints

- 不修改 PP-ShiTuV2、ALIKED、LightGlue、模型权重、512 输入节点或现有全局/局部融合阈值。
- 干扰区域继续只过滤模板侧局部特征；不新增查询图掩码或 PP-ShiTuV2 全局特征掩码。
- 旧版没有标注字段的工件库，以及当前已保存的 `interference_groups` 清单，都必须可恢复。
- 所有标注修改必须带 `operation_id` 和 `base_revision`；过期修订不得覆盖新数据。
- 待复核草稿不得部分进入预测；预测始终使用最近一次完整发布的有效标注修订。
- 类型删除必须按稳定 `group_id` 定位，并从未过滤的原始局部特征重新生成剩余掩码缓存。
- Qt 使用中文文案；状态不能仅以颜色区分，必须同时显示文字。
- 仅修改本功能涉及的文件；不整理或提交工作区中无关的既有改动。
- 所有生产代码均按 TDD 顺序实施：先写失败测试、确认失败原因、做最小实现、回归通过、再提交。

---

## File Structure

### Python domain and protocol

- Modify: `src/workpiece_library.py` — 版本化标注文档、预期修订检查和兼容读取。
- Modify: `src/interference_masks.py` — 有效掩码映射、递推诊断字段和关键点统计纯函数。
- Modify: `src/orientation_classifier.py` — 从原始局部特征准备候选过滤缓存，不直接发布。
- Modify: `src/workpiece_catalog.py` — 完整标注快照、草稿/生效版本提交、类型启停/删除和缓存原子发布。
- Modify: `src/template_evolution.py` — 递推来源审计、重新递推和接受/修正/拒绝/不存在动作。
- Modify: `src/orientation_tcp_service.py` — 查询快照、修改命令、参数校验和稳定错误码。

### Qt application

- Create: `qt_app/annotationcanvas.h`
- Create: `qt_app/annotationcanvas.cpp` — 原图坐标多矩形显示、选择、拖移、增加和删除。
- Create: `qt_app/annotationmanager.h`
- Create: `qt_app/annotationmanager.cpp` — 类型列表、模板状态矩阵、预览、诊断和操作信号。
- Modify: `qt_app/annotationeditor.h`
- Modify: `qt_app/annotationeditor.cpp` — 复用画布编辑已有多矩形并保护未保存修改。
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp` — 串行请求编排、弹窗生命周期和错误恢复。
- Modify: `qt_app/workpiece_orientation.pro`

### Tests and verification

- Modify: `tests/test_workpiece_library.py`
- Modify: `tests/test_interference_masks.py`
- Modify: `tests/test_workpiece_catalog.py`
- Modify: `tests/test_template_evolution.py`
- Modify: `tests/test_orientation_tcp_service.py`
- Create: `qt_app/tests/test_annotationmanager.cpp`
- Create: `qt_app/tests/test_annotationmanager.pro`
- Modify: `qt_app/tests/test_mainwindow.cpp`
- Modify: `qt_app/tests/test_mainwindow.pro`
- Modify: `docs/verification/workpiece-lifecycle-learning-mask-results.md`

---

### Task 1: Versioned annotation document and optimistic revision guard

**Files:**
- Modify: `src/workpiece_library.py:521`
- Modify: `tests/test_workpiece_library.py`

**Interfaces:**
- Produces: `StaleWorkpieceRevisionError(WorkpieceLibraryError)`.
- Produces: `WorkpieceLibrary.get_annotation_document(workpiece_id: str) -> dict[str, Any]`.
- Produces: `WorkpieceLibrary.replace_annotation_document(workpiece_id: str, draft_groups: list[dict], *, expected_revision: int, active_groups: list[dict] | None = None) -> WorkpieceRecord`.
- Document keys are exactly `revision`, `annotation_revision`, `active_annotation_revision`, `draft_groups`, and `active_groups`.
- `active_groups=None` preserves the previous active groups/revision; a list publishes that list and advances `active_annotation_revision` to the new annotation revision.

- [ ] **Step 1: Write compatibility and stale-write tests**

Add tests that first create a legacy manifest through `register`, then directly add only `interference_groups` and verify compatibility:

```python
def test_annotation_document_reads_legacy_groups_as_draft_and_safe_groups_as_active(tmp_path):
    library, record = create_library_with_workpiece(tmp_path)
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["interference_groups"] = [
        {"group_id": "active", "enabled": True,
         "propagation": {"state": "active"}, "annotations": []},
        {"group_id": "review", "enabled": True,
         "propagation": {"state": "needs_review"}, "annotations": []},
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    document = library.get_annotation_document(record.id)

    assert [item["group_id"] for item in document["draft_groups"]] == ["active", "review"]
    assert [item["group_id"] for item in document["active_groups"]] == ["active"]
```

Add a second test that calls `replace_annotation_document` with a stale revision and asserts the manifest bytes and in-memory record revision do not change.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_workpiece_library.py -k "annotation_document or stale_annotation" -vv
```

Expected: FAIL because `get_annotation_document`, `replace_annotation_document`, and `StaleWorkpieceRevisionError` do not exist.

- [ ] **Step 3: Implement legacy normalization and atomic manifest replacement**

Implement these exact compatibility rules:

```python
draft_groups = manifest.get("interference_groups", [])
active_groups = manifest.get("active_interference_groups")
if active_groups is None:
    active_groups = [
        group for group in draft_groups
        if group.get("enabled", True)
        and group.get("propagation", {}).get("state") == "active"
    ]
```

`replace_annotation_document` must compare `expected_revision` with `record.revision` before writing, deep-copy caller data, write `manifest.json.tmp`, replace `manifest.json`, then update `_records`. It writes schema version at least 2 plus:

```python
manifest["interference_groups"] = draft_groups
manifest["active_interference_groups"] = next_active_groups
manifest["annotation_revision"] = previous_annotation_revision + 1
manifest["active_annotation_revision"] = (
    manifest["annotation_revision"] if active_groups is not None
    else previous_active_annotation_revision
)
```

- [ ] **Step 4: Run library tests and verify GREEN**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_workpiece_library.py -vv
```

Expected: all workpiece-library tests PASS, including unchanged 5+5 and dynamic-template compatibility tests.

- [ ] **Step 5: Commit only Task 1 files**

```powershell
git add src/workpiece_library.py tests/test_workpiece_library.py
git commit -m "feat: version interference annotation documents"
```

---

### Task 2: Candidate mask cache and coherent annotation snapshot

**Files:**
- Modify: `src/interference_masks.py`
- Modify: `src/orientation_classifier.py:193`
- Modify: `src/workpiece_catalog.py:121`
- Modify: `tests/test_interference_masks.py`
- Modify: `tests/test_workpiece_catalog.py`

**Interfaces:**
- Consumes: Task 1 `get_annotation_document` and `replace_annotation_document`.
- Produces: `build_active_mask_map(front_count: int, back_count: int, groups: Sequence[dict]) -> dict[str, list[list[dict]]]`.
- Produces: `OrientationClassifier.prepare_template_masks(workpiece_id: str, ignored_regions: dict[str, Any]) -> tuple[TemplateCache, dict[str, list[dict[str, float | int]]]]`.
- Produces: `WorkpieceCatalog.get_annotation_snapshot(workpiece_id: str) -> dict[str, Any]`.
- Produces: `WorkpieceCatalog.commit_annotation_document(workpiece_id: str, draft_groups: list[dict], *, expected_revision: int, operation_id: str, active_groups: list[dict] | None = None) -> dict[str, Any]`.
- Snapshot schema is fixed as:

```python
{
    "workpiece_id": str,
    "revision": int,
    "annotation_revision": int,
    "active_annotation_revision": int,
    "templates": [{
        "template_id": "front:00.png",
        "orientation": "front",
        "index": 0,
        "preview_path": str,
        "width": int,
        "height": int,
        "readable": bool,
        "mask_effect": {"keypoints_before": int, "keypoints_after": int, "remaining_ratio": float},
    }],
    "groups": [{
        "group_id": str,
        "name": str,
        "enabled": bool,
        "draft_state": str,
        "active_state": str,
        "summary": dict,
        "targets": list[dict],
    }],
}
```

- [ ] **Step 1: Write pure mask-map and cache-preparation tests**

Add a pure test proving disabled, rejected, absent and needs-review annotations do not enter the active mask map. Add a classifier/catalog fake that contains raw keypoints and assert preparation returns a new cache plus exact before/after counts while the currently published cache object remains unchanged.

```python
assert statistics["front"][0] == {
    "keypoints_before": 3,
    "keypoints_after": 2,
    "remaining_ratio": pytest.approx(2 / 3),
}
assert classifier.get_template_cache("m7") is original_cache
```

- [ ] **Step 2: Write snapshot and publication rollback tests**

Add `test_annotation_snapshot_contains_templates_status_and_active_revision` and `test_failed_manifest_commit_leaves_published_cache_unchanged`. The latter injects a library double whose `replace_annotation_document` raises `OSError` after cache preparation and asserts the classifier cache is still the original object.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_interference_masks.py tests\test_workpiece_catalog.py -k "mask_map or prepare_template or annotation_snapshot or manifest_commit" -vv
```

Expected: FAIL on the missing functions and methods, without failures from unrelated lifecycle tests.

- [ ] **Step 4: Implement pure candidate cache preparation**

Refactor current `set_template_masks` so both paths use one pure helper. `prepare_template_masks` must always filter from `raw_local_features or local_features`, calculate keypoint counts, and return a candidate `TemplateCache`; it must not assign `_template_caches[workpiece_id]`. Keep `set_template_masks` as a compatibility wrapper that prepares then calls `set_template_cache`.

- [ ] **Step 5: Implement catalog snapshot and two-phase publication**

`commit_annotation_document` performs this order under the catalog lock:

1. Read and verify the current document revision.
2. If `active_groups` is a list, build masks and a candidate cache from that explicit list; if it is `None`, keep the current active groups/cache.
3. Persist the new annotation document with Task 1.
4. Publish the already prepared candidate through `set_template_cache`.
5. Return `get_annotation_snapshot` for the new revision.

The publication branch must remain explicit:

```python
candidate_cache = None
if active_groups is not None:
    masks = build_active_mask_map(len(record.front_images), len(record.back_images), active_groups)
    candidate_cache, mask_statistics = classifier.prepare_template_masks(workpiece_id, masks)
updated = library.replace_annotation_document(
    workpiece_id,
    draft_groups,
    expected_revision=expected_revision,
    active_groups=active_groups,
)
if candidate_cache is not None:
    classifier.set_template_cache(workpiece_id, candidate_cache)
```

Use `template_id = f"{orientation}:{path.name}"`. For an unreadable preview, return `readable=False`, width/height zero and preserve the template row instead of failing the entire query.

- [ ] **Step 6: Add catalog recovery of active masks**

Add `WorkpieceCatalog.recover() -> list[WorkpieceRecord]`. It delegates initial feature building to `library.recover`, publishes every recovered cache, reads `active_groups`, prepares its masks and republishes the filtered candidate. This is the only startup route later used by the TCP runtime.

- [ ] **Step 7: Run focused and existing catalog tests**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_interference_masks.py tests\test_workpiece_catalog.py -vv
```

Expected: all tests PASS; snapshot queries do not mutate cache or revision.

- [ ] **Step 8: Commit only Task 2 files**

```powershell
git add src/interference_masks.py src/orientation_classifier.py src/workpiece_catalog.py tests/test_interference_masks.py tests/test_workpiece_catalog.py
git commit -m "feat: publish coherent annotation snapshots"
```

---

### Task 3: Auditable propagation and review actions

**Files:**
- Modify: `src/interference_masks.py:108`
- Modify: `src/template_evolution.py:166`
- Modify: `tests/test_interference_masks.py`
- Modify: `tests/test_template_evolution.py`

**Interfaces:**
- Consumes: Task 2 `commit_annotation_document` and `get_annotation_snapshot`.
- Changes: `TemplateEvolution.get_annotations(workpiece_id: str) -> dict[str, Any]` now returns the complete snapshot.
- Changes: `TemplateEvolution.save_annotations(workpiece_id: str, groups: list[dict], *, expected_revision: int, operation_id: str) -> dict[str, Any]`.
- Produces: `TemplateEvolution.set_group_enabled(workpiece_id: str, group_id: str, enabled: bool, *, expected_revision: int, operation_id: str) -> dict[str, Any]`.
- Produces: `TemplateEvolution.delete_group(workpiece_id: str, group_id: str, *, expected_revision: int, operation_id: str) -> dict[str, Any]`.
- Produces: `AnnotationGroupNotFoundError` and `InvalidAnnotationReviewError`.
- Review uses the existing `save_annotations` interface. A group patch carries one annotation with `template_id`, `review_action` and `regions`; `review_action` is one of `accept`, `correct`, `reject`, `absent`, `repropagate`.

- [ ] **Step 1: Write diagnostic contract tests**

Extend the two-source propagation tests to assert a successful automatic annotation contains:

```python
diagnostics = automatic["diagnostics"]
assert diagnostics["reason_code"] == "sources_agree"
assert diagnostics["source_template_ids"] == ["front:00.png", "front:01.png"]
assert diagnostics["attempted_source_count"] == 2
assert diagnostics["successful_projection_count"] == 2
assert 0.0 <= diagnostics["confidence"] <= 1.0
assert diagnostics["max_spread_px"] == pytest.approx(0.0)
assert diagnostics["area_ratio"] > 0.0
```

Add disagreement and unreadable-target tests asserting stable reason codes `source_disagreement` and `target_unreadable`.

- [ ] **Step 2: Write group management and review tests**

Cover all observable actions:

- disabling preserves draft annotations but removes the group from active masks;
- enabling a complete group republishes it, while enabling an unresolved group leaves the active revision unchanged;
- deleting one `group_id` preserves the other group and removes deleted masks;
- `accept` and `correct` create trusted annotations;
- `reject` stores a proposal digest and does not publish the draft;
- `absent` resolves that template without a rectangle;
- `repropagate` removes stale automatic candidates before recomputing;
- stale `expected_revision` leaves document and cache unchanged;
- repeated `operation_id` returns the same snapshot without another revision increment.

- [ ] **Step 3: Run evolution tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_template_evolution.py tests\test_interference_masks.py -k "diagnostic or group or review or repropagate" -vv
```

Expected: FAIL because the new signatures, actions and diagnostic keys do not exist.

- [ ] **Step 4: Make geometry results machine-readable**

Update `resolve_propagated_region` so every branch returns `reason_code`, `reason`, `confidence`, `max_spread_px` and `area_ratio`. Preserve the existing 3-pixel correspondence tolerance and 50% maximum area rule; do not introduce a new fusion or recognition threshold.

- [ ] **Step 5: Rework propagation into deterministic replacement**

Before propagating a group, retain only manual/accepted/corrected/absent/rejected records and remove its previous automatic records. For each target, collect projection records as:

```python
{
    "source_template_id": source_template_id,
    "region_index": region_index,
    "projected_region": projected,
}
```

Write `targets` for every template, including manual, automatic active, automatic needs-review, absent, rejected and unresolved states. A group is publishable only when every enabled target is active or confirmed absent.

- [ ] **Step 6: Implement review and type lifecycle methods**

All methods merge by stable `group_id`, use `template_id` to find the target, then call the same propagation-and-commit path. `save_annotations` detects and consumes `review_action` rather than persisting that transport-only field. `delete_group` removes the group independently from the current draft list and current active list, then passes both explicit lists to `commit_annotation_document`; this prevents an unrelated unresolved draft from becoming active. `set_group_enabled(False)` preserves records in the draft but publishes an explicit active list with that group disabled. `set_group_enabled(True)` reruns propagation and passes `active_groups=None` while incomplete, or the complete draft list when publishable.

- [ ] **Step 7: Run the complete evolution suite**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_template_evolution.py tests\test_interference_masks.py tests\test_workpiece_catalog.py -vv
```

Expected: all tests PASS; existing asynchronous confirmation-job tests remain unchanged.

- [ ] **Step 8: Commit only Task 3 files**

```powershell
git add src/interference_masks.py src/template_evolution.py tests/test_interference_masks.py tests/test_template_evolution.py
git commit -m "feat: audit and review mask propagation"
```

---

### Task 4: TCP annotation management contract

**Files:**
- Modify: `src/orientation_tcp_service.py:276`
- Modify: `tests/test_orientation_tcp_service.py`

**Interfaces:**
- Consumes: Task 3 evolution methods.
- `get_workpiece_annotations` response contains `annotations: <snapshot>`.
- `save_workpiece_annotations` requires `workpiece_id`, `groups`, `base_revision`, `operation_id`.
- New command `set_workpiece_annotation_group_enabled` requires `workpiece_id`, `group_id`, `enabled`, `base_revision`, `operation_id`.
- New command `delete_workpiece_annotation_group` requires `workpiece_id`, `group_id`, `base_revision`, `operation_id`.
- Stable error mappings: `STALE_WORKPIECE_REVISION`, `ANNOTATION_GROUP_NOT_FOUND`, `INVALID_ANNOTATION_REVIEW`, `INVALID_MASK`.

- [ ] **Step 1: Extend `FakeEvolution` with exact protocol-facing signatures**

Record every call and return a snapshot carrying revision 3. Do not put group behavior in this fake; adapter tests assert translation only. Review patches use `save_workpiece_annotations`, not a separate command.

- [ ] **Step 2: Write dispatcher tests for query and each mutation**

For deletion, assert exact translation:

```python
response = dispatcher.dispatch({
    "version": 1,
    "request_id": "delete-group-1",
    "command": "delete_workpiece_annotation_group",
    "workpiece_id": "m7",
    "group_id": "glare",
    "base_revision": 2,
    "operation_id": "op-delete-group-1",
})
assert response["ok"] is True
assert response["annotations"]["revision"] == 3
assert evolution.delete_calls == [("m7", "glare", 2, "op-delete-group-1")]
```

Add invalid-type cases for boolean `base_revision`, missing operation IDs, unknown `review_action` values and non-array regions inside a save patch.

- [ ] **Step 3: Run dispatcher tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py -k "annotation" -vv
```

Expected: FAIL because the dispatcher currently returns only `groups` and has no management commands.

- [ ] **Step 4: Implement additive command routing and stable errors**

Use `type(base_revision) is int` so JSON booleans are rejected. Return mutation snapshots under `annotations`. Catch the three new domain errors before the generic `TemplateEvolutionError` branch.

- [ ] **Step 5: Route startup recovery through `WorkpieceCatalog.recover`**

Replace the direct `library.recover`/`classifier.set_template_cache` loop in runtime initialization with the catalog recovery method from Task 2, so active annotation masks survive backend restart.

- [ ] **Step 6: Run protocol and integration regression**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py tests\test_orientation_service_integration.py -vv
```

Expected: all tests PASS; `hello`, `list_workpieces`, `register`, `predict`, progress and shutdown responses remain protocol v1 compatible.

- [ ] **Step 7: Commit only Task 4 files**

```powershell
git add src/orientation_tcp_service.py tests/test_orientation_tcp_service.py
git commit -m "feat: expose annotation management protocol"
```

---

### Task 5: Reusable native-coordinate multi-region canvas

**Files:**
- Create: `qt_app/annotationcanvas.h`
- Create: `qt_app/annotationcanvas.cpp`
- Modify: `qt_app/annotationeditor.h`
- Modify: `qt_app/annotationeditor.cpp`
- Create: `qt_app/tests/test_annotationmanager.cpp`
- Create: `qt_app/tests/test_annotationmanager.pro`

**Interfaces:**
- Produces:

```cpp
struct AnnotationRegionView {
    QRectF rect;
    QString provenance; // manual, automatic
    QString status;     // active, needs_review
    bool selected = false;
};

class AnnotationCanvas : public QWidget {
    Q_OBJECT
public:
    void setImage(const QImage &image);
    void setRegions(const QList<AnnotationRegionView> &regions);
    QList<AnnotationRegionView> regions() const;
    void setEditable(bool editable);
    int selectedRegionIndex() const;
public slots:
    void deleteSelectedRegion();
signals:
    void regionsChanged();
    void selectionChanged(int index);
};
```

- Changes: `AnnotationEditorDialog` receives snapshot/group/template context and returns all native regions, not a single `QRectF`:

```cpp
explicit AnnotationEditorDialog(const QJsonObject &snapshot,
                                const QString &groupId,
                                const QString &templateId,
                                QWidget *parent = nullptr);
QJsonObject annotationPatch() const;
bool hasUnsavedChanges() const;
void setUnsavedPromptHandler(std::function<QMessageBox::StandardButton()> handler);
```

- [ ] **Step 1: Write canvas coordinate, overlay and edit tests**

In `test_annotationmanager.cpp`, use a 200×100 image in a 400×400 canvas and verify aspect-fit padding maps a drag from display `(100, 100)` to native `(0, 0)` and `(300, 300)` to `(200, 100)`. Add tests for two regions, selection, dragging one region, deletion, and style-independent provenance/status storage.

- [ ] **Step 2: Write editor dirty-state tests**

Construct the editor with two saved rectangles. Verify both load, adding a third sets `hasUnsavedChanges()`, and an injected prompt handler can return Save, Discard or Cancel when switching templates.

- [ ] **Step 3: Build the Qt test and verify RED**

Run from a Visual Studio developer environment:

```powershell
New-Item -ItemType Directory -Force qt_app\build-test-annotationmanager | Out-Null
Set-Location qt_app\build-test-annotationmanager
E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe ..\tests\test_annotationmanager.pro CONFIG+=debug
E:\QT\5.14\Tools\QtCreator\bin\jom.exe
$env:QT_QPA_PLATFORM = 'offscreen'
.\debug\test_annotationmanager.exe -txt
```

Expected: compilation FAIL because `annotationcanvas.h` and the expanded editor interfaces do not exist.

- [ ] **Step 4: Move canvas responsibility out of the editor**

Implement aspect-fit native/display transforms once in `AnnotationCanvas`. Render manual active as green solid, automatic active as blue solid, needs-review as orange dashed, and selected as a thicker outline. Keep provenance/status text available to the parent; color is supplemental.

- [ ] **Step 5: Implement multi-region interaction**

Dragging empty image space creates a new normalized rectangle. Clicking a rectangle selects it. Dragging a selected rectangle moves it while clamping to image bounds. Delete removes only the selected rectangle. Regions smaller than one native pixel are discarded on mouse release.

- [ ] **Step 6: Update the editor and unsaved-change guard**

The editor loads existing regions for the selected template, permits multiple rectangles, returns a complete annotation patch, and calls its prompt handler before template/group/window changes. Use a default handler backed by `QMessageBox`; tests inject a deterministic handler.

- [ ] **Step 7: Rebuild and run the canvas/editor test**

Run the Step 3 build and test commands again.

Expected: `test_annotationmanager.exe -txt` reports all canvas/editor cases PASS.

- [ ] **Step 8: Commit only Task 5 files**

```powershell
git add qt_app/annotationcanvas.h qt_app/annotationcanvas.cpp qt_app/annotationeditor.h qt_app/annotationeditor.cpp qt_app/tests/test_annotationmanager.cpp qt_app/tests/test_annotationmanager.pro
git commit -m "feat: edit multiple interference regions"
```

---

### Task 6: Annotation manager dialog and propagation audit UI

**Files:**
- Create: `qt_app/annotationmanager.h`
- Create: `qt_app/annotationmanager.cpp`
- Modify: `qt_app/tests/test_annotationmanager.cpp`
- Modify: `qt_app/tests/test_annotationmanager.pro`
- Modify: `qt_app/workpiece_orientation.pro`

**Interfaces:**
- Consumes: Task 4 snapshot schema and Task 5 canvas/editor.
- Produces:

```cpp
class AnnotationManagerDialog : public QDialog {
    Q_OBJECT
public:
    explicit AnnotationManagerDialog(QWidget *parent = nullptr);
    void setSnapshot(const QJsonObject &snapshot);
    void setBusy(bool busy);
    void setDeleteConfirmationHandler(std::function<bool(const QString &name,
                                                         int affectedTemplates)> handler);
    QString workpieceId() const;
    int baseRevision() const;
signals:
    void refreshRequested();
    void saveRequested(const QJsonArray &groups, int baseRevision);
    void setEnabledRequested(const QString &groupId, bool enabled, int baseRevision);
    void deleteRequested(const QString &groupId, int baseRevision);
    void reviewRequested(const QString &groupId, const QString &templateId,
                         const QString &action, const QJsonArray &regions, int baseRevision);
};
```

- Object names required by QTest: `annotationGroupList`, `annotationTemplateTable`, `annotationPreviewCanvas`, `annotationDiagnosticsText`, `annotationVersionLabel`, `addAnnotationGroupButton`, `renameAnnotationGroupButton`, `toggleAnnotationGroupButton`, `deleteAnnotationGroupButton`, `acceptAnnotationButton`, `correctAnnotationButton`, `rejectAnnotationButton`, `absentAnnotationButton`, `repropagateAnnotationButton`.

- [ ] **Step 1: Write snapshot-rendering tests**

Create a fixture with one manual active target, one automatic active target, one automatic needs-review target and one absent target. Assert the dialog renders the server-provided summary counts and exact Chinese statuses without recalculating propagation state.

- [ ] **Step 2: Write preview and diagnostics tests**

Select each target row and assert the preview loads its `preview_path`, canvas region count matches the target, and diagnostics text contains source template IDs, confidence, maximum spread, area ratio and keypoint before/after counts. For `readable=false`, assert the visible error text is `模板图片无法读取` and no exception is emitted.

- [ ] **Step 3: Write management signal tests**

Use `QSignalSpy` to verify:

- delete emits stable `group_id` and current revision only after confirmation handler returns true;
- disable/enable emits the opposite current state;
- accept/correct/reject/absent/repropagate emit exact action strings and selected `template_id`;
- mutation controls are disabled while `setBusy(true)` and re-enabled after a refreshed snapshot.

- [ ] **Step 4: Build and run to verify RED**

Use the Task 5 qmake/jom/test commands.

Expected: compilation FAIL because `AnnotationManagerDialog` does not exist.

- [ ] **Step 5: Implement the three-panel manager**

Build the dialog with Qt Widgets in code: group list on the left, target table in the middle, canvas plus read-only diagnostics on the right. Store IDs and full JSON objects in `Qt::UserRole`; visible names are never used as mutation identifiers.

- [ ] **Step 6: Implement state text, overlays and operations**

Render `已生效`, `待复核`, `已停用`, `无有效种子`, and `保存中` from the snapshot. Show both `annotation_revision` and `active_annotation_revision`; when unequal, display `草稿尚未参与预测，当前仍使用修订前稳定掩码`. Add confirmation text with type name and affected template count before delete.

- [ ] **Step 7: Integrate the editor for add/correct operations**

New/edit/correct actions open `AnnotationEditorDialog` with the current snapshot. On acceptance, emit a complete group patch or review correction; the manager never writes files or sends TCP requests itself.

- [ ] **Step 8: Update qmake projects and run dialog tests**

Add the new canvas/manager sources and headers to both `.pro` files, rebuild and run offscreen tests.

Expected: all annotation manager tests PASS.

- [ ] **Step 9: Commit only Task 6 files**

```powershell
git add qt_app/annotationmanager.h qt_app/annotationmanager.cpp qt_app/tests/test_annotationmanager.cpp qt_app/tests/test_annotationmanager.pro qt_app/workpiece_orientation.pro
git commit -m "feat: add interference annotation manager"
```

---

### Task 7: MainWindow request orchestration and stale-state recovery

**Files:**
- Modify: `qt_app/mainwindow.h:48`
- Modify: `qt_app/mainwindow.cpp:70,402,610,716,733`
- Modify: `qt_app/tests/test_mainwindow.cpp`
- Modify: `qt_app/tests/test_mainwindow.pro`

**Interfaces:**
- Consumes: Task 6 manager signals and Task 4 TCP commands.
- Changes: existing `annotationEditorButton` remains the object name for test compatibility, but visible text becomes `管理干扰标注` and click first queries the selected workpiece snapshot.
- Produces private slots: `openAnnotationManager()`, `saveAnnotationGroups(...)`, `setAnnotationGroupEnabled(...)`, `deleteAnnotationGroup(...)`, `reviewAnnotation(...)`.
- Produces member: `QPointer<AnnotationManagerDialog> annotationManagerDialog_`.

- [ ] **Step 1: Extend the fake Qt TCP server**

Make the test server respond to `get_workpiece_annotations`, save, enable, delete and review commands with increasing revisions and a complete annotation snapshot. Record all requests for exact assertion.

- [ ] **Step 2: Write MainWindow integration tests**

Cover:

- the management button is disabled with no selected workpiece, during loading and while the client is Busy;
- click sends one `get_workpiece_annotations` request and opens the manager after the response;
- save/enable/delete signals map to their exact commands and include `base_revision` plus a nonempty `operation_id`;
- review signals are encoded as an annotation patch with `review_action` and sent through `save_workpiece_annotations`;
- successful mutation refreshes the manager snapshot;
- `STALE_WORKPIECE_REVISION` displays `标注已在其他操作中更新，已重新加载` and performs a fresh query without retrying the mutation;
- changing selected workpiece closes or invalidates the old manager;
- backend disconnect disables manager mutations and keeps the main window responsive.

- [ ] **Step 3: Build MainWindow test and verify RED**

From `qt_app/build-test-mainwindow-dynamic`, rerun qmake so new sources enter the Makefile, then jom and:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\debug\test_mainwindow.exe -txt
```

Expected: FAIL because the button still opens the one-shot editor and no manager command flow exists.

- [ ] **Step 4: Replace direct one-shot save with snapshot-first manager flow**

`openAnnotationManager` validates selected workpiece/client readiness, sets `pendingCommand_` to `get_workpiece_annotations`, sends one request and shows a loading message. Construct or update the dialog only after a successful snapshot response.

- [ ] **Step 5: Wire mutation signals through the single outstanding request rule**

Every handler sets the manager busy, sends exactly one request, and includes the dialog revision. On success, immediately send one refresh only after `BackendClient` returns Ready; on failure, clear busy and preserve unsaved editor state where applicable.

- [ ] **Step 6: Handle stale revision and workpiece invalidation**

For `STALE_WORKPIECE_REVISION`, do not replay the old edit. Show the conflict message and fetch a new snapshot. Close the manager if the selected workpiece changes, is recycled or disappears from the refreshed list.

- [ ] **Step 7: Run Qt MainWindow and manager regression**

Run both offscreen executables:

```powershell
qt_app\build-test-mainwindow-dynamic\debug\test_mainwindow.exe -txt
qt_app\build-test-annotationmanager\debug\test_annotationmanager.exe -txt
```

Expected: all Qt tests PASS with exit code 0.

- [ ] **Step 8: Commit only Task 7 files**

```powershell
git add qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/tests/test_mainwindow.cpp qt_app/tests/test_mainwindow.pro
git commit -m "feat: connect annotation manager to backend"
```

---

### Task 8: Full compatibility, release build and verification record

**Files:**
- Modify: `docs/verification/workpiece-lifecycle-learning-mask-results.md`
- Verify only: all production and test files from Tasks 1–7.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: reproducible verification record containing commands, pass counts, Qt build result, real-model availability and known limitations.

- [ ] **Step 1: Run Python syntax checks**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m py_compile src\workpiece_library.py src\interference_masks.py src\orientation_classifier.py src\workpiece_catalog.py src\template_evolution.py src\orientation_tcp_service.py
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Run the complete Python test suite**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q
```

Expected: all available tests PASS. If the environment lacks `lightglue` or configured model assets, only tests explicitly marked for those optional assets may be skipped; collection errors are failures.

- [ ] **Step 3: Run all Qt tests offscreen**

Rebuild after qmake, then run:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
qt_app\build-test-backendclient-dynamic\debug\test_backendclient.exe -txt
qt_app\build-test-backendprocessmanager\debug\test_backendprocessmanager.exe -txt
qt_app\build-test-mainwindow-dynamic\debug\test_mainwindow.exe -txt
qt_app\build-test-annotationmanager\debug\test_annotationmanager.exe -txt
```

Expected: every executable exits 0 and reports no failed QTest cases.

- [ ] **Step 4: Build the Qt 5.14.2 release application**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_qt5.ps1
```

Expected: `Qt build succeeded` and `qt_app\build-release\release\workpiece_orientation.exe` exists.

- [ ] **Step 5: Run a local protocol smoke test**

Start the configured backend, then verify this sequence against a disposable test workpiece: query snapshot, create two trusted annotations, inspect propagation, disable, re-enable, delete the group, restart backend and query again. Expected: revisions increase monotonically; stale revision is rejected; deleted masks do not reappear after restart.

- [ ] **Step 6: Run configured real-model mask verification when dependencies are available**

First run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -c "import lightglue; print('lightglue available')"
```

If this succeeds, run the existing project-image mask verification and record before/after local keypoint statistics while confirming global embeddings and fusion constants are byte-for-byte/config-for-config unchanged. If it fails, record `BLOCKED: lightglue unavailable in shitu environment`; do not report the real-model verification as passed.

- [ ] **Step 7: Update the verification record**

Record exact Python pass/skip counts, each Qt executable result, release executable path, smoke-test revisions, and real-model result in `docs/verification/workpiece-lifecycle-learning-mask-results.md`.

- [ ] **Step 8: Check diff scope and whitespace**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; every changed path maps to a task in this plan. Existing unrelated dirty paths remain untouched and unstaged.

- [ ] **Step 9: Commit the verification record**

```powershell
git add docs/verification/workpiece-lifecycle-learning-mask-results.md
git commit -m "test: verify annotation management workflow"
```

---

## Execution Checkpoints

- After Task 2: review manifest/cache atomicity and legacy recovery before adding more actions.
- After Task 4: review the complete TCP JSON contract and stable errors before Qt consumes it.
- After Task 6: visually inspect the manager with one active, one pending and one disabled group before wiring MainWindow.
- After Task 8: compare `git diff` with the approved design and confirm no model or fusion constant changed.

## Non-Goals During Execution

- Do not train or fine-tune any model.
- Do not add semantic glare/shadow classification.
- Do not add query-side or global-feature masking.
- Do not add multi-client support or remote annotation access.
- Do not recalibrate recognition thresholds while implementing the management UI.
- Do not refactor unrelated MainWindow, template-ingestion or lifecycle behavior.
