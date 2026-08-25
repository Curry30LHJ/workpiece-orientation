# Runtime Snapshot Stability Design

**Date:** 2026-08-22  
**Status:** Approved in conversation; awaiting written-spec review  
**Scope:** Runtime snapshot isolation, priority model execution, confirmed-template append transactions, Qt error separation, and backend restart semantics

## Context

The desktop application currently routes online prediction, workpiece registration, confirmed-template ingestion, and geometry validation through one TCP client, one `WorkpieceCatalog`, and shared Paddle/Torch model instances. `WorkpieceCatalog.predict()` and `WorkpieceCatalog.append_templates()` hold the same catalog lock, while append rebuilds every template feature and refreshes geometry state. A confirmed-template job can therefore block online prediction for the duration of a full cache rebuild.

The current lock graph also contains opposite acquisition orders: catalog operations can call geometry-profile operations while holding the catalog lock, and geometry-profile operations can call back into the catalog while holding the geometry lock. This is a potential deadlock.

Qt currently sends all `BackendClient::requestFailed` signals to `BackendProcessManager`. Most domain failures are consequently treated as service failures, which clears inspection and batch state and disables the geometry dialog even though the TCP service remains ready.

Finally, `WorkpieceLibrary.append_templates()` creates a replacement directory containing only the two template directories and `manifest.json`. It does not copy `geometry_masks/`, so the first confirmed-template append after publishing a geometry profile can remove the profile, revision history, and previews.

## Goals

1. Online prediction continues against the last published workpiece revision while a candidate revision is built.
2. An online request receives model priority no later than the end of the background operation's current template step.
3. Paddle and Torch model execution is serialized; no true concurrent GPU execution is introduced.
4. A candidate revision becomes visible only after all files and features are prepared and its base revision still matches.
5. Failed, cancelled, or stale candidates never change the active disk record or active in-memory cache.
6. Confirmed-template append preserves all workpiece sidecar data, including `geometry_masks/` and its revision history.
7. A restart after disk commit but before job completion can reconcile the job as completed without adding the same images twice.
8. Domain command failures leave the backend ready and preserve user workflow state.
9. Restart and reconnect behavior reflects whether the Qt session owns the Python process.
10. Existing workpiece libraries, including old 5+5 libraries and arbitrary unequal template counts, remain readable.

## Non-goals

This change does not modify PP-ShiTu, ALIKED, LightGlue, image preprocessing, score fusion, thresholds, or model weights. It does not correct max-score template-count bias, redesign geometry publication, extend the persistent cache fingerprint, prune geometry-job history, or change the 224 by 224 recognition resize. Those are separate changes because they affect accuracy or a second persistent state machine.

## Chosen Architecture

The system will use an immutable active snapshot per workpiece and a cooperatively preemptible, single-executor model gate.

```text
Online request
  -> capture ActiveWorkpieceSnapshot(vN) under a short catalog lock
  -> release catalog lock
  -> acquire high-priority model lease
  -> predict entirely from vN

Confirmed-template job
  -> capture ActiveWorkpieceSnapshot(vN)
  -> prepare a complete staged workpiece directory
  -> build candidate features one template step at a time under low-priority leases
  -> create PreparedTemplateUpdate(vN -> vN+1)
  -> validate base revision under a short catalog lock
  -> atomically commit the staged directory and swap the active snapshot
```

A prediction that starts before publication finishes with `vN`. A prediction that captures its snapshot after publication uses `vN+1`. No request observes a mixture of revisions.

## Runtime Types and Ownership

### `ActiveWorkpieceSnapshot`

Add a frozen value object owned by `WorkpieceCatalog`:

```python
@dataclass(frozen=True)
class ActiveWorkpieceSnapshot:
    record: WorkpieceRecord
    cache: TemplateCache
```

`record.revision` is the snapshot version. `WorkpieceCatalog` owns a mapping from workpiece ID to the active snapshot. Capturing the object requires the catalog lock only long enough to read one mapping entry.

`OrientationClassifier.predict_with_cache(cache, image_path)` performs prediction from the supplied immutable cache. Production prediction must not look up a mutable cache by workpiece ID after the snapshot has been captured. Existing ID-based cache methods may remain temporarily for compatibility tests and untouched legacy paths, but the prediction and confirmed-template paths must use explicit cache arguments.

### `PreparedTemplateUpdate`

Add an owned transaction object:

```python
@dataclass
class PreparedTemplateUpdate:
    operation_id: str
    workpiece_id: str
    base_revision: int
    staging_root: Path
    staged_record: WorkpieceRecord
    candidate_cache: TemplateCache
    item_digests: tuple[str, ...]
```

The object owns `staging_root` until commit or abort. Commit consumes it. Abort removes it. Repeated abort is safe. A prepared update never mutates `WorkpieceLibrary._records`.

### `PriorityModelGate`

Use one process-wide gate for PP-ShiTu, ALIKED, and LightGlue execution. Its public interface is `run_online(action: Callable[[], T]) -> T` and `run_background_step(action: Callable[[], T]) -> T`.

The gate permits exactly one action at a time. Waiting online actions take precedence over a new background step. A background step is one template's global and local feature extraction, or one bounded geometry-validation unit. An active step is not interrupted halfway through; background code yields before beginning the next step.

This gate replaces model-level lock layering on the changed paths. Catalog, geometry-profile, and model-gate locks must never be held while acquiring one another. In particular, no geometry-profile call is made while the catalog lock is held.

## Prediction Flow

1. `WorkpieceCatalog.predict()` captures the active snapshot under the catalog lock.
2. It releases the lock before reading the query image or invoking any model.
3. `OrientationClassifier.predict_with_cache()` acquires an online model lease and uses only the captured cache.
4. The result includes the snapshot's library revision for diagnostics.
5. Publication during prediction changes only the snapshot used by later requests.

The classifier's feature objects are treated as immutable after construction. Publication swaps the containing cache reference rather than modifying lists or arrays in an active cache.

## Confirmed-Template Preparation

`WorkpieceLibrary.prepare_append()` performs the slow work outside the catalog lock:

1. Validate the new images and their content digests against the captured base record.
2. Clone the complete active workpiece root into a unique `.staging-*` directory.
3. Replace the staged `0/` and `1/` directories with the combined old and new templates.
4. Preserve all other files and directories, including `geometry_masks/`, immutable geometry revisions, previews, and future sidecars.
5. Remove the copied `.template_cache.pkl` because its signature belongs to the base revision.
6. Write a staged schema-version-3 manifest with the new template counts, candidate revision, and `last_template_update` metadata:

```json
{
  "operation_id": "8f3d55a1912d41d1a2599ecbf154634a",
  "base_revision": 7,
  "target_revision": 8,
  "item_digests": ["3f83c0a4e8397f8bfbe64bd4a90d99cb78b2e622d3dc7e1841ce617d54d1082f"]
}
```

7. Build the candidate base cache from the staged template paths. Each image extraction uses one background model step.
8. If an active geometry profile exists, update only the staged `geometry_masks/profile.json` library-revision pointer, then build the effective candidate cache from the explicit base cache and staged profile. Immutable revision files, preview files, rule definitions, and active-revision pointers remain unchanged. This work also occurs outside catalog and geometry-profile locks.
9. Persist the unfiltered base portion of the candidate cache into the staged workpiece root before returning the prepared update. The prepared snapshot holds the effective cache, which may be the geometry view or the raw base view.

The append path does not silently omit any selected or confirmed image. Existing arbitrary front/back counts remain valid, and the counts may differ.

## Atomic Commit and Conflict Handling

`WorkpieceCatalog.commit_prepared_append()` performs the following while holding the catalog lock:

1. Compare the active snapshot revision with `prepared.base_revision`.
2. If they differ, release the lock, abort the staged directory, and report a stable stale-revision result to the job layer.
3. Rename the current workpiece root to a unique `.backup-*` path.
4. Rename the prepared staging root to the formal workpiece path.
5. Construct the committed `WorkpieceRecord` using formal paths.
6. Atomically replace the catalog's active snapshot reference.
7. Update the library's active record mapping.
8. Release the lock and remove the backup.

Existing predictions can finish from the captured old cache even after its directory has been replaced because the cache is memory-resident and immutable. New predictions use the committed snapshot.

If a filesystem operation fails before the formal root is installed, the backup is restored. If the process terminates after the formal root is installed, startup recovery treats the formal root as authoritative and removes the backup, matching the existing recovery convention.

Only one prepared update per workpiece may be in the building state. A second confirmation can coalesce into a queued successor job, but it cannot modify an in-progress prepared update.

## Restart Reconciliation

The template-evolution job ID is the append transaction's `operation_id`. On startup or before retrying a job that was previously `building`:

1. Load the current workpiece manifest.
2. Compare `last_template_update.operation_id` with the job ID.
3. Verify target revision and all item digests.
4. If they match, mark the job completed and clean its staged confirmation files.
5. If they do not match, return the job to queued state only when its base revision is still current.
6. If the base revision changed for another reason, mark the job stale and retain a retryable diagnostic rather than appending blindly.

Malformed job storage is quarantined and reported without changing a valid active workpiece snapshot. It must not turn an otherwise recoverable model service into `MODEL_LOAD_FAILED`.

## Qt Error Semantics

`BackendClient` will expose two failure channels:

```cpp
void commandFailed(const QString &command,
                   const QString &code,
                   const QString &message);
void transportFailed(const QString &code,
                     const QString &message);
```

An error response with a matching request ID emits `commandFailed` and leaves the client in `Ready`. Socket, timeout, handshake, and protocol failures emit `transportFailed` and transition the client out of `Ready`.

`BackendProcessManager` listens only to transport and handshake lifecycle signals. It never converts a domain command failure into `backendUnavailable`.

`MainWindow` handles `commandFailed` using the command supplied by the signal instead of relying on a mutable `pendingCommand_` that another slot may clear first. A domain failure ends the affected busy indicator and preserves template selections, geometry drafts, selected workpiece, inspection evidence, and completed batch rows.

A transport failure during a mutating operation is displayed as an unknown outcome. Qt does not automatically resend that mutation. Template-evolution reconciliation supplies the final status for confirmed-template jobs.

## Backend Process Lifecycle

Restart behavior depends on ownership:

- For a Python process launched by the current Qt session: request graceful shutdown, wait for process exit, call `terminate()` after a grace deadline, call `kill()` only after a second deadline, wait for the port to become available, and then launch exactly one replacement process.
- For an externally owned service: disconnect and reconnect only. The UI labels the action as reconnect rather than restart.
- A valid service that reports `loading` is never treated as evidence that Qt should start another process.
- The accepted handshake socket has a finite timeout so a connection that sends no complete hello message cannot reserve the service indefinitely.

Shutdown and restart are explicit state-machine transitions. `start()` is not called synchronously immediately after `terminate()`.

Geometry validation workers are constructed without starting their thread, catalog recovery completes first, and workers start only after the runtime has published a ready snapshot. Persisted queued jobs therefore cannot call back into a half-recovered catalog.

## Failure Semantics

| Failure | Active prediction state | Candidate state | UI state |
|---|---|---|---|
| New image invalid or duplicate | Unchanged | Rejected | Command error; backend remains ready |
| Feature extraction fails | Unchanged | Failed and abortable | Job failure; prediction remains available |
| Base revision changes | Newer snapshot remains active | Stale | Retry offered after refresh |
| Staging commit fails | Old snapshot remains active | Failed; backup restored | Job failure |
| Process exits before commit | Old snapshot recovered | Staging discarded | Job may retry |
| Process exits after commit | New snapshot recovered | Reconciled as completed | No duplicate append |
| Domain command fails | Unchanged | Command-specific | Draft and batch state preserved |
| TCP connection fails | Last committed disk state | Outcome may be unknown | Backend unavailable; no blind retry |

## Testing Strategy

All behavior changes are implemented test-first.

### Python concurrency tests

- Block candidate feature construction with a test barrier, start `predict`, and prove prediction completes from the old cache before the barrier is released.
- Record gate entry order and prove maximum model concurrency is one.
- Queue an online request while one background step is active and prove it runs before the next background step.
- Prepare two candidates from the same revision and prove only the first can commit.
- Start a prediction on `vN`, publish `vN+1`, and prove the in-flight result reports `vN` while the next result reports `vN+1`.

### Persistence tests

- Append templates to a workpiece containing a geometry profile, immutable revisions, and previews. Prove immutable revision and preview hashes are unchanged, the staged `profile.json` points at the new library revision, and the active rule definition is unchanged.
- Inject failure before and after each directory rename and prove recovery yields exactly one complete old or new version.
- Persist a committed manifest while leaving the evolution job in `building`, restart, and prove it reconciles to completed without adding templates again.
- Verify malformed job JSON is quarantined while valid workpiece snapshots still recover.
- Recover legacy manifests without `last_template_update` and arbitrary unequal template counts.

### Qt tests

- Return a domain error and prove `commandFailed` fires, `backendUnavailable` does not, and the client remains ready.
- Prove a geometry command failure preserves dialog content and releases only that dialog's busy state.
- Prove a batch command failure preserves completed rows.
- Prove owned restart waits for the old process-finished signal before launching again.
- Prove external reconnect never calls the process launcher.
- Hold a handshake connection without sending a line and prove another client can connect after the timeout.

### Regression and performance checks

- Run the complete project Python test suite in the `shitu` environment.
- Rebuild and run Qt 5.14.2 tests from current sources rather than relying on existing binaries.
- Run real-model smoke checks on existing M1, M2, and M7 libraries.
- Measure prediction latency with 1+1, 5+10, 10+15, and 35+35 templates while a background append is active. Report foreground latency separately from total background completion time.

## Acceptance Criteria

1. A test-controlled long append cannot hold the catalog lock long enough to block prediction.
2. A prediction uses one immutable revision from start to finish.
3. Online model work is served before the next background template step and total model concurrency remains one.
4. Failed or stale candidates never alter active disk or memory state.
5. Confirmed-template append preserves immutable geometry revisions and previews byte-for-byte, preserves active rule semantics, and updates only the profile's library-revision pointer.
6. Restart reconciliation cannot append the same confirmed images twice.
7. Domain command errors cannot emit `backendUnavailable` or clear unrelated workflow state.
8. Owned restart launches no replacement before the old process has exited; external reconnect launches no process.
9. Legacy and unequal-count libraries remain compatible.
10. Model files, model code, fusion policy, and thresholds have no behavioral diff in this change.

## Follow-up Changes

After this change is verified, subsequent independent designs should address, in order: geometry publication atomicity and job retention; template-count-neutral score aggregation and template governance; cache model fingerprinting and input normalization; then Qt workflow simplification and operational telemetry.
