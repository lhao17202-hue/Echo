"""Filesystem-backed registry for workspace-local skills."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on active environment
    yaml = None

logger = logging.getLogger("echo.skills")
ValidationLevel = Literal["warning", "error"]


@dataclass(frozen=True)
class Skill:
    """A discovered skill manifest."""

    name: str
    description: str
    path: Path
    content: str


@dataclass(frozen=True)
class ValidationIssue:
    """Validation issue discovered while scanning skills."""

    path: str
    level: ValidationLevel
    message: str


@dataclass(frozen=True)
class _SkillManifest:
    """Normalized SKILL.md manifest used by scan() and validate()."""

    name: str
    description: str
    path: Path
    content: str
    parse_error: str = ""


class SkillRegistry:
    """Registry for filesystem-based skills under a skills/ directory."""

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        self._issues: list[ValidationIssue] = []

    def scan(self) -> "SkillRegistry":
        """Scan skills_dir for */SKILL.md manifests."""
        self._skills.clear()
        manifests, discovery_issues = self._discover_manifests(report_missing=False)
        self._issues = [*discovery_issues, *self._validate_manifests(manifests)]

        for manifest in manifests:
            if not self._is_valid_name(manifest.name):
                logger.warning("Invalid skill name skipped: %s", manifest.name)
                continue
            if manifest.name in self._skills:
                logger.warning("Duplicate skill name skipped: %s", manifest.name)
                continue
            self._skills[manifest.name] = Skill(
                name=manifest.name,
                description=manifest.description,
                path=manifest.path,
                content=manifest.content,
            )
        return self

    def get(self, name: str) -> Skill | None:
        """Return a registered skill by exact name."""
        if not self._is_valid_name(name):
            return None
        return self._skills.get(name)

    def load_content(self, name: str) -> str | None:
        """Return full SKILL.md content by exact registered name."""
        skill = self.get(name)
        return skill.content if skill else None

    def list_catalog(self) -> str:
        """Render lightweight skill catalog for the system prompt."""
        return "\n".join(
            f"- **{skill.name}**: {skill.description}"
            for skill in self._skills.values()
        )

    def names(self) -> list[str]:
        """Return registered skill names."""
        return list(self._skills.keys())

    def validate(self) -> list[ValidationIssue]:
        """Return structured validation issues for the skills directory."""
        manifests, discovery_issues = self._discover_manifests(report_missing=True)
        self._issues = [*discovery_issues, *self._validate_manifests(manifests)]
        return list(self._issues)

    def _discover_manifests(self, report_missing: bool) -> tuple[list[_SkillManifest], list[ValidationIssue]]:
        manifests: list[_SkillManifest] = []
        issues: list[ValidationIssue] = []
        if not self.skills_dir.exists():
            return manifests, issues

        for child in sorted(self.skills_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.is_symlink():
                issues.append(ValidationIssue(
                    path=str(child),
                    level="error",
                    message="Symlink skill directories are not allowed",
                ))
                continue
            if not child.is_dir():
                continue
            manifest = child / "SKILL.md"
            if not manifest.exists():
                if report_missing:
                    issues.append(ValidationIssue(
                        path=str(manifest),
                        level="warning",
                        message="Missing SKILL.md",
                    ))
                continue
            if manifest.is_symlink():
                issues.append(ValidationIssue(
                    path=str(manifest),
                    level="error",
                    message="Symlink SKILL.md manifests are not allowed",
                ))
                continue
            if not manifest.is_file():
                issues.append(ValidationIssue(
                    path=str(manifest),
                    level="error",
                    message="SKILL.md is not a regular file",
                ))
                continue
            if not self._is_within_skills_dir(manifest):
                issues.append(ValidationIssue(
                    path=str(manifest),
                    level="error",
                    message="SKILL.md resolves outside the skills directory",
                ))
                continue
            raw = self._read_text(manifest)
            manifests.append(self._normalize_manifest(child, manifest, raw))
        return manifests, issues

    def _validate_manifests(self, manifests: list[_SkillManifest]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        seen: dict[str, Path] = {}
        for manifest in manifests:
            if manifest.parse_error:
                issues.append(ValidationIssue(
                    path=str(manifest.path),
                    level="error",
                    message=f"Invalid frontmatter: {manifest.parse_error}",
                ))
            if not self._is_valid_name(manifest.name):
                issues.append(ValidationIssue(
                    path=str(manifest.path),
                    level="error",
                    message=f"Invalid skill name: {manifest.name or '<empty>'}",
                ))
                continue
            if not manifest.description:
                issues.append(ValidationIssue(
                    path=str(manifest.path),
                    level="error",
                    message="Missing description",
                ))
            if manifest.name in seen:
                issues.append(ValidationIssue(
                    path=str(manifest.path),
                    level="error",
                    message=f"Duplicate skill name: {manifest.name} also used by {seen[manifest.name]}",
                ))
            else:
                seen[manifest.name] = manifest.path
        return issues

    def _normalize_manifest(self, child: Path, manifest: Path, raw: str) -> _SkillManifest:
        meta, body, parse_error = self._parse_frontmatter(raw)
        name = str(meta.get("name") or child.name).strip()
        description = str(
            meta.get("description") or self._fallback_description(body or raw)
        ).strip()
        return _SkillManifest(
            name=name,
            description=description,
            path=manifest,
            content=raw,
            parse_error=parse_error,
        )

    def _is_within_skills_dir(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.skills_dir.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str, str]:
        """Return (metadata, body, parse_error)."""
        if not text.startswith("---"):
            return {}, text, ""
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text, "unterminated frontmatter"
        try:
            meta = SkillRegistry._load_yaml_mapping(parts[1])
        except Exception as exc:  # pragma: no cover - defensive
            return {}, parts[2].strip(), str(exc)
        return meta, parts[2].strip(), ""

    @staticmethod
    def _load_yaml_mapping(text: str) -> dict:
        if yaml is not None:
            meta = yaml.safe_load(text) or {}
            return meta if isinstance(meta, dict) else {}

        meta: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                continue
            meta[key] = value.strip().strip('"\'')
        return meta

    @staticmethod
    def _fallback_description(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            return stripped
        return ""

    @staticmethod
    def _is_valid_name(name: str) -> bool:
        if not name or not name.strip():
            return False
        if "/" in name or "\\" in name or ".." in name:
            return False
        if re.match(r"^[A-Za-z]:", name):
            return False
        return True
