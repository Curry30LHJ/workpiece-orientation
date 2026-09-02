"""Durable, explicit human-confirmed template ingestion jobs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import shutil
import threading
import time
import uuid
from typing import Callable

from src.image_io import read_color_image
from src.interference_masks import resolve_propagated_region
from src.orientation_classifier import ImageUnreadableError, PropagationModelError
from src.native_pp_client import NativePPError
from src.workpiece_catalog import WorkpieceCatalog
from src.workpiece_library import (
    IMAGE_EXTENSIONS,
    InvalidTemplateSetError,
    WorkpieceLibraryError,
)


LOGGER = logging.getLogger(__name__)


class TemplateEvolutionError(RuntimeError):
    """Base evolution error."""


class DuplicateTemplateError(TemplateEvolutionError):
    """Raised when decoded image content is already present."""


class InvalidConfirmationError(TemplateEvolutionError):
    """Raised when an explicit confirmation is malformed."""


class StaleEvolutionError(TemplateEvolutionError):
    """Raised when the workpiece changed after a confirmation was captured."""


class AnnotationGroupNotFoundError(TemplateEvolutionError):
    """Raised when an annotation group id is not present in a workpiece."""


class InvalidAnnotationReviewError(TemplateEvolutionError):
    """Raised when an annotation review action or target is invalid."""


class TemplateEvolution:
    """Single-worker durable queue with immutable active-cache publication."""

    COALESCE_SECONDS = 3.0
    MAX_PROJECTION_FAILURES = 3

    def __init__(self, catalog: WorkpieceCatalog, storage_dir: Path, *, clock: Callable[[], float] | None = None,
                 duration_clock: Callable[[], float] | None = None,
                 start_worker: bool = True, geometry_profiles=None):
        self.catalog = catalog
        self.geometry_profiles = geometry_profiles or getattr(catalog, "geometry_profiles", None)
        self.storage_dir = Path(storage_dir)
        self.staging_dir = self.storage_dir / "staging"
        self.jobs_path = self.storage_dir / "jobs.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock or time.time
        self._duration_clock = duration_clock or time.perf_counter
        self._duration_started: dict[str, float] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._jobs: dict[str, dict] = {}
        self._operation_results: dict[str, str] = {}
        self._load()
        self._stop = False
        self._worker = None
        if start_worker:
            self.start()

    def start(self) -> None:
        """Start the queue worker once, after the runtime snapshot is ready."""
        with self._condition:
            if self._worker is not None and self._worker.is_alive():
                return
            if self._stop:
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="template-evolution",
                daemon=True,
            )
            self._worker.start()

    @staticmethod
    def _fingerprint(path: Path) -> str:
        image = read_color_image(path)
        if image is None:
            raise InvalidConfirmationError(f"Unable to read confirmation image: {path}")
        digest = hashlib.sha256()
        digest.update(str(image.shape).encode("ascii"))
        digest.update(str(image.dtype).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()

    def _load(self) -> None:
        if not self.jobs_path.exists():
            return
        try:
            payload = json.loads(self.jobs_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("jobs document must be an object")
            operations = payload.get("operations", {})
            jobs = payload.get("jobs", [])
            if not isinstance(operations, dict) or not isinstance(jobs, list):
                raise ValueError("jobs document has an invalid schema")
            loaded_jobs: dict[str, dict] = {}
            for item in jobs:
                if not isinstance(item, dict) or not isinstance(item.get("job_id"), str) or not item["job_id"]:
                    raise ValueError("jobs document contains an invalid job")
                if item["job_id"] in loaded_jobs:
                    raise ValueError("jobs document contains duplicate job ids")
                loaded_jobs[item["job_id"]] = item
            self._operation_results = {str(key): str(value) for key, value in operations.items()}
            self._jobs = loaded_jobs
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            quarantine = self.storage_dir / f"jobs.corrupt-{uuid.uuid4().hex}.json"
            self.jobs_path.replace(quarantine)
            self._operation_results = {}
            self._jobs = {}
            LOGGER.error("Quarantined invalid template evolution jobs at %s: %s", quarantine, exc)
            return
        changed = False
        for job in self._jobs.values():
            state = job.get("state")
            total = len(job.get("items", []))
            defaults = {
                "phase": "active" if state == "completed" else "queued",
                "completed": total if state == "completed" else 0,
                "total": total,
                "progress": 100 if state == "completed" else 0,
                "recovery_detail": None,
                "started_at": None,
                "finished_at": None,
                "elapsed_ms": None,
                "error_code": None,
                "error_phase": None,
                "retryable": None,
            }
            for key, value in defaults.items():
                if key not in job:
                    job[key] = value
                    changed = True
            if state == "building":
                self._reset_queued_progress(job)
                job["error"] = None
                job["recovery_detail"] = "上次缓存构建因后端重启中断，任务已重新排队并将自动重试"
                changed = True
        if changed:
            self._persist()

    def _persist(self) -> None:
        temp = self.jobs_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"jobs": list(self._jobs.values()), "operations": self._operation_results},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.jobs_path)

    def _snapshot(self, job: dict) -> dict:
        snapshot = deepcopy(job)
        if snapshot.get("state") == "building":
            duration_started = self._duration_started.get(str(snapshot.get("job_id")))
            if duration_started is not None:
                snapshot["elapsed_ms"] = self._duration_ms(
                    duration_started, float(self._duration_clock())
                )
        return snapshot

    def _existing_digests(self, workpiece_id: str) -> set[str]:
        record = self.catalog.get(workpiece_id)
        return {self._fingerprint(path) for path in (*record.front_images, *record.back_images)}

    def submit_confirmation(self, workpiece_id: str, orientation: str, image_path: Path, *, operation_id: str) -> dict:
        if orientation not in {"front", "back"}:
            raise InvalidConfirmationError("orientation must be front or back")
        if not isinstance(operation_id, str) or not operation_id:
            raise InvalidConfirmationError("operation_id must be a non-empty string")
        with self._condition:
            if operation_id in self._operation_results:
                return self._snapshot(self._jobs[self._operation_results[operation_id]])
            source = Path(image_path)
            if source.suffix.lower() not in IMAGE_EXTENSIONS or not source.is_file():
                raise InvalidConfirmationError(f"Invalid confirmation image: {source}")
            digest = self._fingerprint(source)
            if digest in self._existing_digests(workpiece_id):
                raise DuplicateTemplateError(f"Duplicate template content: {source}")
            for job in self._jobs.values():
                if job["workpiece_id"] != workpiece_id or job["state"] in {"completed", "cancelled"}:
                    continue
                if any(item["digest"] == digest for item in job["items"]):
                    raise DuplicateTemplateError(f"Duplicate queued template content: {source}")
            record = self.catalog.get(workpiece_id)
            now = float(self._clock())
            target_job = None
            for job in reversed(list(self._jobs.values())):
                if (job["workpiece_id"] == workpiece_id and job["state"] == "queued"
                        and now - float(job["last_submitted_at"]) <= self.COALESCE_SECONDS):
                    target_job = job
                    break
            if target_job is None:
                predecessor = next(
                    (job for job in reversed(list(self._jobs.values()))
                     if job["workpiece_id"] == workpiece_id and job["state"] == "building"),
                    None,
                )
                job_id = uuid.uuid4().hex
                target_job = {
                    "job_id": job_id,
                    "workpiece_id": workpiece_id,
                    "base_revision": record.revision,
                    "state": "queued",
                    "phase": "queued",
                    "completed": 0,
                    "total": 0,
                    "progress": 0,
                    "recovery_detail": None,
                    "started_at": None,
                    "finished_at": None,
                    "elapsed_ms": None,
                    "warnings": [],
                    "error": None,
                    "last_submitted_at": now,
                    "items": [],
                }
                if predecessor is not None:
                    target_job["predecessor_job_id"] = predecessor["job_id"]
                self._jobs[job_id] = target_job
            target_job["last_submitted_at"] = now
            suffix = source.suffix.lower() or ".png"
            staged = self.staging_dir / f"{target_job['job_id']}-{len(target_job['items'])}{suffix}"
            shutil.copy2(source, staged)
            target_job["items"].append({"orientation": orientation, "path": str(staged), "digest": digest})
            target_job["total"] = len(target_job["items"])
            self._operation_results[operation_id] = target_job["job_id"]
            self._persist()
            self._condition.notify_all()
            return self._snapshot(target_job)

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            return self._snapshot(self._jobs[job_id])

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._snapshot(job) for job in self._jobs.values()]

    @staticmethod
    def _template_id(orientation: str, path: Path) -> str:
        return f"{orientation}:{path.name}"

    @staticmethod
    def _annotation_key(annotation: dict) -> tuple[str, object]:
        template_id = annotation.get("template_id")
        if isinstance(template_id, str) and template_id:
            return "template_id", template_id
        return "index", (annotation.get("orientation"), annotation.get("index"))

    def _template_location(self, template_id: str, record) -> tuple[str, int] | None:
        for orientation, paths in (("front", record.front_images), ("back", record.back_images)):
            for index, path in enumerate(paths):
                if self._template_id(orientation, path) == template_id:
                    return orientation, index
        return None

    @staticmethod
    def _proposal_digest(annotation: dict) -> str:
        payload = json.dumps(
            {
                "template_id": annotation.get("template_id"),
                "orientation": annotation.get("orientation"),
                "index": annotation.get("index"),
                "regions": annotation.get("regions", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _existing_annotation_key(self, annotations: dict[tuple[str, object], dict],
                                 template_id: str, record) -> tuple[str, object]:
        key = ("template_id", template_id)
        if key in annotations:
            return key
        for candidate_key, candidate in annotations.items():
            if candidate.get("template_id") == template_id:
                return candidate_key
            orientation = candidate.get("orientation")
            index = candidate.get("index")
            paths = record.front_images if orientation == "front" else record.back_images if orientation == "back" else ()
            if type(index) is int and 0 <= index < len(paths):
                if self._template_id(orientation, paths[index]) == template_id:
                    return candidate_key
        return key

    def _apply_review_action(self, annotations: dict[tuple[str, object], dict], patch: dict, record) -> None:
        action = patch.get("review_action")
        if action not in {"accept", "correct", "reject", "absent", "repropagate"}:
            raise InvalidAnnotationReviewError(f"unknown annotation review action: {action}")
        template_id = patch.get("template_id")
        if not isinstance(template_id, str) or ":" not in template_id:
            orientation = patch.get("orientation")
            index = patch.get("index")
            paths = record.front_images if orientation == "front" else record.back_images if orientation == "back" else ()
            if type(index) is not int or not 0 <= index < len(paths):
                raise InvalidAnnotationReviewError("review target must identify a valid template")
            template_id = self._template_id(orientation, paths[index])
            patch = dict(patch)
            patch["template_id"] = template_id
        key = self._existing_annotation_key(annotations, template_id, record)
        if action == "repropagate":
            # Repropagation discards old automatic proposals in _propagate_group.
            # Operator decisions (including rejected/confirmed-absent entries) must
            # remain in the draft; only trusted entries are used as propagation sources.
            return
        current = deepcopy(annotations.get(key, patch))
        current["template_id"] = template_id
        location = self._template_location(template_id, record)
        if location is not None:
            current["orientation"], current["index"] = location
        if action in {"correct", "absent"}:
            regions = patch.get("regions", [])
            if not isinstance(regions, list):
                raise InvalidAnnotationReviewError("review regions must be an array")
            if action == "correct":
                for region in regions:
                    from src.interference_masks import validate_region
                    validate_region(region)
                current.update({"regions": deepcopy(regions), "trusted": True,
                                "status": "active", "provenance": "corrected"})
            else:
                current.update({"regions": [], "trusted": True,
                                "status": "confirmed_absent", "provenance": "manual"})
        elif action == "accept":
            current.update({"trusted": True, "status": "active", "provenance": "accepted"})
        elif action == "reject":
            current.update({"trusted": False, "status": "rejected", "provenance": "review",
                            "proposal_digest": self._proposal_digest(current)})
        current.pop("review_action", None)
        annotations[key] = current

    def _groups_publishable(self, groups: list[dict]) -> bool:
        for group in groups:
            if not group.get("enabled", True):
                continue
            propagation = group.get("propagation", {})
            if propagation.get("state") != "active" or propagation.get("unresolved"):
                return False
            if any(
                annotation.get("status") not in {"active", "confirmed_absent"}
                for annotation in group.get("annotations", [])
                if annotation.get("provenance") == "automatic"
            ):
                return False
        return True

    def save_annotations(
        self,
        workpiece_id: str,
        groups: list[dict],
        *,
        expected_revision: int | None = None,
        operation_id: str,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict:
        if not isinstance(groups, list) or not groups:
            raise InvalidConfirmationError("at least one interference group is required")
        with self._lock:
            record = self.catalog.get(workpiece_id)
            if expected_revision is None:
                expected_revision = record.revision
            document = self.catalog.get_annotation_document(workpiece_id)
            combined: dict[str, dict] = {
                str(item.get("group_id", item.get("name", ""))): deepcopy(item)
                for item in document["draft_groups"]
            }
            for item in groups:
                key = str(item.get("group_id", item.get("name", "")))
                current = combined.get(key, {"group_id": key, "name": item.get("name", key), "annotations": []})
                by_target = {self._annotation_key(a): deepcopy(a) for a in current.get("annotations", [])}
                for annotation in item.get("annotations", []):
                    if annotation.get("review_action"):
                        self._apply_review_action(by_target, annotation, record)
                    else:
                        by_target[self._annotation_key(annotation)] = deepcopy(annotation)
                current.update({name: value for name, value in item.items() if name != "annotations"})
                current["annotations"] = list(by_target.values())
                combined[key] = current
            prepared_groups = []
            for raw_group in combined.values():
                group = deepcopy(raw_group)
                self._propagate_group(
                    group,
                    record,
                    progress_callback=progress_callback,
                    workpiece_id=workpiece_id,
                    operation_id=operation_id,
                )
                prepared_groups.append(group)
            active_groups = prepared_groups if self._groups_publishable(prepared_groups) else None
            return self.catalog.commit_annotation_document(
                workpiece_id,
                prepared_groups,
                expected_revision=expected_revision,
                operation_id=operation_id,
                active_groups=active_groups,
            )

    def get_annotations(self, workpiece_id: str) -> dict:
        return self.catalog.get_annotation_snapshot(workpiece_id)

    def delete_group(self, workpiece_id: str, group_id: str, *, expected_revision: int,
                     operation_id: str) -> dict:
        with self._lock:
            document = self.catalog.get_annotation_document(workpiece_id)
            draft_groups = [group for group in document["draft_groups"]
                            if str(group.get("group_id", group.get("name", ""))) != group_id]
            if len(draft_groups) == len(document["draft_groups"]):
                raise AnnotationGroupNotFoundError(f"Unknown annotation group: {group_id}")
            active_groups = [group for group in document["active_groups"]
                             if str(group.get("group_id", group.get("name", ""))) != group_id]
            return self.catalog.commit_annotation_document(
                workpiece_id,
                draft_groups,
                expected_revision=expected_revision,
                operation_id=operation_id,
                active_groups=active_groups,
            )

    def set_group_enabled(self, workpiece_id: str, group_id: str, enabled: bool, *, expected_revision: int,
                          operation_id: str) -> dict:
        if type(enabled) is not bool:
            raise InvalidAnnotationReviewError("enabled must be boolean")
        with self._lock:
            document = self.catalog.get_annotation_document(workpiece_id)
            groups = deepcopy(document["draft_groups"])
            target = next((group for group in groups
                           if str(group.get("group_id", group.get("name", ""))) == group_id), None)
            if target is None:
                raise AnnotationGroupNotFoundError(f"Unknown annotation group: {group_id}")
            target["enabled"] = enabled
            if not enabled:
                active_groups = [group for group in document["active_groups"]
                                 if str(group.get("group_id", group.get("name", ""))) != group_id]
                return self.catalog.commit_annotation_document(
                    workpiece_id, groups, expected_revision=expected_revision,
                    operation_id=operation_id, active_groups=active_groups,
                )
            record = self.catalog.get(workpiece_id)
            prepared = []
            for group in groups:
                item = deepcopy(group)
                self._propagate_group(
                    item,
                    record,
                    workpiece_id=workpiece_id,
                    operation_id=operation_id,
                )
                prepared.append(item)
            active_groups = prepared if self._groups_publishable(prepared) else None
            return self.catalog.commit_annotation_document(
                workpiece_id, prepared, expected_revision=expected_revision,
                operation_id=operation_id, active_groups=active_groups,
            )

    def _propagate_group(
        self,
        group: dict,
        record,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        *,
        workpiece_id: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        retained = [item for item in group.get("annotations", []) if item.get("provenance") != "automatic"]
        group["annotations"] = retained
        seeds = [item for item in retained
                 if item.get("trusted", False) and item.get("status", "active") not in {"rejected", "confirmed_absent"}]
        if len(seeds) < 2:
            group["propagation"] = {
                "state": "needs_review",
                "reason_code": "insufficient_sources",
                "reason": "at least two trusted sources required",
                "automatic_count": 0,
                "unresolved_count": 0,
            }
            return
        targets = [("front", index, path) for index, path in enumerate(record.front_images)]
        targets.extend(("back", index, path) for index, path in enumerate(record.back_images))
        total = len(targets)
        if progress_callback is not None:
            progress_callback({"phase": "propagating_annotations", "completed": 0, "total": total})
        resolved_keys = {
            (item.get("orientation"), item.get("index"))
            for item in retained
            if item.get("status", "active") in {"active", "confirmed_absent", "rejected"}
        }
        unresolved = []
        automatic = []
        projection_failure_count = 0
        project = getattr(self.catalog.classifier, "project_region_between_templates", None)
        for completed, (orientation, index, target_path) in enumerate(targets, start=1):
            target_template_id = self._template_id(orientation, target_path)
            projections = []
            projection_sources = []
            failed_source_template_ids = []
            try:
                if (orientation, index) in resolved_keys:
                    continue
                for seed in seeds:
                    if not callable(project):
                        continue
                    source_index = seed.get("index")
                    source_paths = record.front_images if seed.get("orientation") == "front" else record.back_images
                    if not isinstance(source_index, int) or not 0 <= source_index < len(source_paths):
                        continue
                    source_template_id = self._template_id(seed["orientation"], source_paths[source_index])
                    for region in seed.get("regions", []):
                        try:
                            projected = project(source_paths[source_index], target_path, region)
                        except PropagationModelError:
                            raise
                        except Exception as exc:
                            projection_failure_count += 1
                            failed_source_template_ids.append(source_template_id)
                            LOGGER.warning(
                                "annotation propagation pair failed operation_id=%s workpiece_id=%s group_id=%s source=%s target=%s exception=%s failure_count=%s",
                                operation_id,
                                workpiece_id,
                                group.get("group_id", group.get("name", "")),
                                source_template_id,
                                self._template_id(orientation, target_path),
                                type(exc).__name__,
                                projection_failure_count,
                                exc_info=True,
                            )
                            if projection_failure_count >= self.MAX_PROJECTION_FAILURES:
                                LOGGER.error(
                                    "annotation propagation aborted operation_id=%s workpiece_id=%s group_id=%s exception=%s failure_count=%s",
                                    operation_id,
                                    workpiece_id,
                                    group.get("group_id", group.get("name", "")),
                                    type(exc).__name__,
                                    projection_failure_count,
                                    exc_info=True,
                                )
                                raise PropagationModelError(
                                    "递推失败次数过多，已中止本次保存，请检查后端日志"
                                ) from exc
                            continue
                        if projected is not None:
                            projections.append(projected)
                            projection_sources.append(source_template_id)
                if len(projections) < 2:
                    unresolved.append({
                        "orientation": orientation,
                        "index": index,
                        "template_id": target_template_id,
                        "reason_code": "projection_failed" if failed_source_template_ids else "insufficient_projections",
                        "reason": "one or more source projections failed" if failed_source_template_ids
                        else "no two valid correspondences",
                        "attempted_source_count": len(seeds),
                        "successful_projection_count": len(projections),
                        "projection_failure_count": len(failed_source_template_ids),
                        "failed_source_template_ids": list(dict.fromkeys(failed_source_template_ids)),
                    })
                    continue
                target_image = read_color_image(target_path)
                if target_image is None:
                    unresolved.append({
                        "orientation": orientation,
                        "index": index,
                        "template_id": target_template_id,
                        "reason_code": "target_unreadable",
                        "reason": "target image unreadable",
                        "attempted_source_count": len(seeds),
                        "successful_projection_count": len(projections),
                        "projection_failure_count": len(failed_source_template_ids),
                        "failed_source_template_ids": list(dict.fromkeys(failed_source_template_ids)),
                    })
                    continue
                resolved = resolve_propagated_region(projections, target_image.shape[:2])
                diagnostics = dict(resolved)
                diagnostics.update({
                    "source_template_ids": list(dict.fromkeys(projection_sources)),
                    "attempted_source_count": len(seeds),
                    "successful_projection_count": len(projections),
                    "projection_failure_count": len(failed_source_template_ids),
                    "failed_source_template_ids": list(dict.fromkeys(failed_source_template_ids)),
                })
                if resolved.get("status") != "active":
                    unresolved.append({
                        "orientation": orientation,
                        "index": index,
                        "template_id": target_template_id,
                        **diagnostics,
                    })
                    if resolved.get("region") is not None:
                        automatic.append({
                            "orientation": orientation,
                            "index": index,
                            "template_id": target_template_id,
                            "regions": [resolved["region"]],
                            "trusted": False,
                            "status": "needs_review",
                            "provenance": "automatic",
                            "diagnostics": diagnostics,
                        })
                    continue
                automatic.append({
                    "orientation": orientation,
                    "index": index,
                    "template_id": target_template_id,
                    "regions": [resolved["region"]],
                    "trusted": False,
                    "status": "active",
                    "provenance": "automatic",
                    "diagnostics": diagnostics,
                })
            except PropagationModelError:
                raise
            except Exception as exc:
                projection_failure_count += 1
                LOGGER.warning(
                    "annotation propagation target failed operation_id=%s workpiece_id=%s group_id=%s target=%s exception=%s failure_count=%s",
                    operation_id,
                    workpiece_id,
                    group.get("group_id", group.get("name", "")),
                    target_template_id,
                    type(exc).__name__,
                    projection_failure_count,
                    exc_info=True,
                )
                if projection_failure_count >= self.MAX_PROJECTION_FAILURES:
                    LOGGER.error(
                        "annotation propagation aborted operation_id=%s workpiece_id=%s group_id=%s target=%s exception=%s failure_count=%s",
                        operation_id,
                        workpiece_id,
                        group.get("group_id", group.get("name", "")),
                        target_template_id,
                        type(exc).__name__,
                        projection_failure_count,
                        exc_info=True,
                    )
                    raise PropagationModelError(
                        "递推失败次数过多，已中止本次保存，请检查后端日志"
                    ) from exc
                unresolved.append({
                    "orientation": orientation,
                    "index": index,
                    "template_id": target_template_id,
                    "reason_code": "projection_failed",
                    "reason": "target projection resolution failed",
                    "attempted_source_count": len(seeds),
                    "successful_projection_count": len(projections),
                    "projection_failure_count": len(failed_source_template_ids) + 1,
                    "failed_source_template_ids": list(dict.fromkeys(failed_source_template_ids)),
                })
            finally:
                if progress_callback is not None:
                    progress_callback({"phase": "propagating_annotations", "completed": completed, "total": total})
        if unresolved:
            group["propagation"] = {
                "state": "needs_review",
                "unresolved": unresolved,
                "automatic_count": sum(1 for item in automatic if item.get("status") == "active"),
                "unresolved_count": len(unresolved),
            }
        else:
            group["propagation"] = {
                "state": "active",
                "automatic_count": len(automatic),
                "unresolved_count": 0,
            }
        group["annotations"] = retained + automatic

    def run_next(self, *, force: bool = True) -> dict | None:
        with self._condition:
            queued = [job for job in self._jobs.values() if job["state"] == "queued"]
            if not queued:
                return None
            job = queued[0]
            if not force and self._clock() - float(job["last_submitted_at"]) < self.COALESCE_SECONDS:
                return None
            job["state"] = "building"
            job["phase"] = "validating"
            job["completed"] = 0
            job["total"] = len(job.get("items", []))
            job["progress"] = 0
            job["recovery_detail"] = None
            job["error"] = None
            job["error_code"] = None
            job["error_phase"] = None
            job["retryable"] = None
            job["started_at"] = float(self._clock())
            job["finished_at"] = None
            job["elapsed_ms"] = 0
            self._duration_started[job["job_id"]] = float(self._duration_clock())
            self._persist()

        def update_progress(event: dict[str, object]) -> None:
            phase = event.get("phase")
            if phase not in {"copying", "features", "committing"}:
                return
            completed = max(0, int(event.get("completed", 0)))
            total = max(0, int(event.get("total", 0)))
            with self._condition:
                current_job = self._jobs.get(job["job_id"])
                if current_job is not job or current_job.get("state") != "building":
                    return
                current_job["phase"] = phase
                current_job["completed"] = completed
                current_job["total"] = total
                current_job["progress"] = 0 if total == 0 else min(99, completed * 100 // total)
                self._update_elapsed(current_job)
                self._persist()
        phase = "validation"
        try:
            current = self.catalog.get(job["workpiece_id"])
            if self._matches_committed_template_update(job, current):
                with self._condition:
                    job["state"] = "completed"
                    job["phase"] = "active"
                    job["completed"] = job.get("total", 0)
                    job["progress"] = 100
                    job["revision"] = current.revision
                    job["error"] = None
                    job["error_code"] = None
                    job["error_phase"] = None
                    job["retryable"] = None
                    job["recovery_detail"] = None
                    self._finish_elapsed(job)
                    self._cleanup_payload(job)
                    self._persist()
                    return self._snapshot(job)
            if current.revision != job["base_revision"]:
                predecessor_id = job.get("predecessor_job_id")
                predecessor = self._jobs.get(predecessor_id) if predecessor_id else None
                if predecessor is None or predecessor.get("state") not in {"completed", "failed", "cancelled"}:
                    raise StaleEvolutionError(
                        "workpiece revision changed without a matching template operation"
                    )
                job["base_revision"] = current.revision
            phase = "geometry_validation"
            geometry_review = self._requires_geometry_review(job, current)
            if geometry_review is not None:
                with self._condition:
                    job["state"] = "needs_review"
                    job["error"] = geometry_review.get("reason", "new template geometry could not be fitted")
                    job["error_code"] = "GEOMETRY_REVIEW_REQUIRED"
                    job["error_phase"] = "geometry_validation"
                    job["retryable"] = True
                    job["review_reason"] = "geometry_mask_low_confidence"
                    job["geometry_review"] = geometry_review
                    self._finish_elapsed(job)
                    self._persist()
                    return self._snapshot(job)
            front = [Path(item["path"]) for item in job["items"] if item["orientation"] == "front"]
            back = [Path(item["path"]) for item in job["items"] if item["orientation"] == "back"]
            phase = "template_cache"
            record, _ = self.catalog.append_templates(
                job["workpiece_id"],
                front,
                back,
                operation_id=job["job_id"],
                progress_callback=update_progress,
                source="confirmed_inspection",
            )
            with self._condition:
                job["state"] = "completed"
                job["phase"] = "active"
                job["completed"] = job.get("total", 0)
                job["progress"] = 100
                job["revision"] = record.revision
                job["error"] = None
                job["error_code"] = None
                job["error_phase"] = None
                job["retryable"] = None
                job["recovery_detail"] = None
                self._finish_elapsed(job)
                self._cleanup_payload(job)
                self._persist()
                return self._snapshot(job)
        except Exception as exc:
            with self._condition:
                job["state"] = "failed"
                job["error"] = str(exc)
                job["error_code"], job["retryable"] = self._classify_failure(exc, phase)
                job["error_phase"] = phase
                LOGGER.exception(
                    "template evolution job failed job_id=%s workpiece_id=%s phase=%s code=%s",
                    job.get("job_id"), job.get("workpiece_id"), phase, job.get("error_code"),
                )
                self._finish_elapsed(job)
                self._persist()
                return self._snapshot(job)

    @staticmethod
    def _matches_committed_template_update(job: dict, record) -> bool:
        """Recognize a disk commit that happened before job completion persisted."""
        try:
            manifest = json.loads((Path(record.root) / "manifest.json").read_text(encoding="utf-8"))
            update = manifest.get("last_template_update")
            if not isinstance(update, dict):
                return False
            expected_digests = {
                str(item["digest"])
                for item in job.get("items", [])
                if isinstance(item, dict) and item.get("digest")
            }
            stored_digests = update.get("item_digests")
            return (
                update.get("operation_id") == job.get("job_id")
                and update.get("base_revision") == job.get("base_revision")
                and update.get("target_revision") == record.revision
                and isinstance(stored_digests, list)
                and set(map(str, stored_digests)) == expected_digests
                and bool(expected_digests)
            )
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def _requires_geometry_review(self, job: dict, record) -> dict | None:
        if self.geometry_profiles is not None:
            for item in job["items"]:
                result = self.geometry_profiles.validate_new_template(
                    job["workpiece_id"], item["orientation"], Path(item["path"])
                )
                if result.get("status") != "active" and result.get("needs_review", True):
                    return result
            return None
        if self._requires_mask_review(job, record):
            return {
                "status": "legacy_archived",
                "needs_review": True,
                "reason": "new template has unresolved interference groups",
            }
        return None

    def _requires_mask_review(self, job: dict, record) -> bool:
        groups = self.catalog.get_annotation_snapshot(job["workpiece_id"])["groups"]
        if not groups:
            return False
        next_index = {"front": len(record.front_images), "back": len(record.back_images)}
        for item in job["items"]:
            orientation = item["orientation"]
            target_index = next_index[orientation]
            next_index[orientation] += 1
            for group in groups:
                if group.get("propagation", {}).get("state") != "active":
                    return True
                resolved = any(
                    annotation.get("orientation") == orientation
                    and annotation.get("index") == target_index
                    and annotation.get("status", "active") == "active"
                    for annotation in group.get("annotations", [])
                )
                if not resolved:
                    return True
        return False

    def action(self, job_id: str, action: str) -> dict:
        with self._condition:
            job = self._jobs[job_id]
            if action == "cancel" and job["state"] in {"queued", "needs_review", "failed"}:
                job["state"] = "cancelled"
                self._cleanup_payload(job)
            elif action == "retry" and job["state"] in {"failed", "needs_review"}:
                self._reset_queued_progress(job)
                self._duration_started.pop(job_id, None)
                job["error"] = None
                job["error_code"] = None
                job["error_phase"] = None
                job["retryable"] = None
                job["recovery_detail"] = None
                job["last_submitted_at"] = float(self._clock())
            elif action == "resolve-review" and job["state"] == "needs_review":
                self._reset_queued_progress(job)
                self._duration_started.pop(job_id, None)
                job["error"] = None
                job["error_code"] = None
                job["error_phase"] = None
                job["retryable"] = None
                job["recovery_detail"] = None
            else:
                raise TemplateEvolutionError(f"job action is not valid for state {job['state']}")
            self._persist()
            self._condition.notify_all()
            return self._snapshot(job)

    def cancel_for_workpiece(self, workpiece_id: str) -> None:
        with self._condition:
            for job in self._jobs.values():
                if job["workpiece_id"] == workpiece_id and job["state"] in {"queued", "needs_review", "failed"}:
                    job["state"] = "cancelled"
                    self._cleanup_payload(job)
            self._persist()
            self._condition.notify_all()

    @staticmethod
    def _reset_queued_progress(job: dict) -> None:
        job["state"] = "queued"
        job["phase"] = "queued"
        job["completed"] = 0
        job["total"] = len(job.get("items", []))
        job["progress"] = 0
        job["started_at"] = None
        job["finished_at"] = None
        job["elapsed_ms"] = None
        job["error_code"] = None
        job["error_phase"] = None
        job["retryable"] = None

    @staticmethod
    def _classify_failure(exc: Exception, phase: str) -> tuple[str, bool]:
        """Return a stable client-facing code without changing the original message."""
        # Native PP-ShiTu failures already carry a protocol-level code.  Keep
        # it intact so the Qt client can tell a missing model/cache from a
        # generic template-cache failure and offer the right recovery action.
        native_code = getattr(exc, "code", None)
        if isinstance(exc, NativePPError) and isinstance(native_code, str) and native_code:
            return native_code, True
        if isinstance(exc, DuplicateTemplateError):
            return "DUPLICATE_TEMPLATE", False
        if isinstance(exc, InvalidConfirmationError):
            return "INVALID_CONFIRMATION", False
        if isinstance(exc, ImageUnreadableError):
            return "IMAGE_UNREADABLE", False
        if isinstance(exc, StaleEvolutionError):
            return "STALE_EVOLUTION", True
        if isinstance(exc, InvalidTemplateSetError):
            return "INVALID_TEMPLATE_SET", False
        if isinstance(exc, PropagationModelError):
            return "MODEL_ERROR", True
        text = str(exc).strip()
        prefixed_code = text.partition(":")[0].strip()
        if prefixed_code.startswith(("NATIVE_PP_", "FAST_CACHE_")):
            return prefixed_code, True
        if isinstance(exc, WorkpieceLibraryError) or phase == "template_cache":
            return "TEMPLATE_CACHE_BUILD_FAILED", True
        return "EVOLUTION_JOB_FAILED", True

    @staticmethod
    def _duration_ms(started_at: float, finished_at: float) -> int:
        return max(0, int(round((finished_at - started_at) * 1000.0)))

    def _update_elapsed(self, job: dict) -> None:
        duration_started = self._duration_started.get(str(job.get("job_id")))
        if duration_started is None:
            return
        job["elapsed_ms"] = self._duration_ms(
            duration_started, float(self._duration_clock())
        )

    def _finish_elapsed(self, job: dict) -> None:
        wall_finished = float(self._clock())
        wall_started = job.get("started_at")
        job["finished_at"] = (
            wall_finished
            if wall_started is None
            else max(float(wall_started), wall_finished)
        )
        self._update_elapsed(job)
        self._duration_started.pop(str(job.get("job_id")), None)

    @staticmethod
    def _cleanup_payload(job: dict) -> None:
        for item in job.get("items", []):
            try:
                Path(item["path"]).unlink(missing_ok=True)
            except OSError:
                pass

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                if self._stop:
                    return
                queued = [job for job in self._jobs.values() if job["state"] == "queued"]
                if not queued:
                    self._condition.wait(timeout=0.5)
                    continue
                wait = max(0.0, self.COALESCE_SECONDS - (self._clock() - queued[0]["last_submitted_at"]))
                if wait > 0:
                    self._condition.wait(timeout=wait)
                    continue
            self.run_next(force=True)

    def shutdown(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2)
