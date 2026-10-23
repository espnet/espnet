"""The rule that decides what a release is called.

It is load-bearing: the milestone title becomes version.txt, version.txt is
what PyPI publishes, and the tag is `v.` plus the same string. A title the
rule lets through and PyPI will not accept is a release that dies at the
upload, which is how the 202604 series went out wrong.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / "doc" / "make_release.py"


def _module():
    # doc/ is not a package, and the script imports github at module level
    pytest.importorskip("github")
    spec = importlib.util.spec_from_file_location("make_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "title, version",
    [
        ("v.202610", "202610"),
        ("v.202610.post1", "202610.post1"),
        ("v.202610.post12", "202610.post12"),
        ("202610", "202610"),  # the prefix is optional
    ],
)
def test_a_release_milestone_names_its_version(title, version):
    assert _module().release_version(title) == version


@pytest.mark.parametrize(
    "title",
    [
        # PyPI will not take this one, and version.txt is what it publishes
        "v.202610-patch1",
        "v.202610patch1",
        "v.2026101",
        "v.20261",
        "v.202610.post",
        "v.202610.dev1",
        # \d would take these; PyPI would not
        "v.２０２６１０",
        "v.202610.post１",
        "Backlog",
        "",
    ],
)
def test_anything_pypi_would_refuse_is_not_a_release(title):
    assert _module().release_version(title) is None


def _pin_files(module, tmp_path, monkeypatch, dockerfile, readme):
    (tmp_path / "Dockerfile").write_text(dockerfile)
    (tmp_path / "README.md").write_text(readme)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "VERSION_PINS",
        [(tmp_path / "Dockerfile", pattern) for _, pattern in module.VERSION_PINS[:1]]
        + [(tmp_path / "README.md", pattern) for _, pattern in module.VERSION_PINS[1:]],
    )


def test_apply_writes_the_release_into_the_docker_pins(tmp_path, monkeypatch):
    module = _module()
    _pin_files(
        module,
        tmp_path,
        monkeypatch,
        "FROM x\nARG ESPNET_VERSION=202610\nRUN pip install espnet\n",
        "docker build --build-arg ESPNET_VERSION=202610 -t espnet-inference .\n",
    )
    assert module.pins_up_to_date("202611", will_apply=False)
    # --apply is what writes them, so their being stale is not a problem there
    assert module.pins_up_to_date("202611", will_apply=True) == []

    module.update_pins("202611")

    assert (tmp_path / "Dockerfile").read_text() == (
        "FROM x\nARG ESPNET_VERSION=202611\nRUN pip install espnet\n"
    )
    assert (
        "--build-arg ESPNET_VERSION=202611 -t" in (tmp_path / "README.md").read_text()
    )
    assert module.pins_up_to_date("202611", will_apply=False) == []


def test_a_pin_the_pattern_no_longer_finds_fails_even_under_apply(
    tmp_path, monkeypatch
):
    module = _module()
    _pin_files(module, tmp_path, monkeypatch, "FROM x\n", "nothing here\n")
    assert len(module.pins_up_to_date("202611", will_apply=True)) == 2


def test_the_docker_pins_name_the_release_in_version_txt():
    module = _module()
    version = module.VERSION_FILE.read_text().strip()
    for path, versions in module.pinned_versions():
        assert versions, f"{path}: VERSION_PINS finds nothing"
        assert set(versions) == {version}, f"{path} says {versions}, not {version}"


def _tags(*names):
    # what repo.get_tags() yields, in the order GitHub listed them for 202611
    return [SimpleNamespace(name=name) for name in names]


GITHUB_TAG_ORDER = [
    "v.202611",
    "v.202610",
    "v.202610.post2",
    "v.202610.post1",
    "v.202609-patch1",
    "untagged-0cfd089fdcf6777f43a3",
    "v.202609",
    "v.202604-patch1",
    "v.202604",
    "v.202402",
    "v.0.10.6",
]


@pytest.mark.parametrize(
    "version, previous",
    [
        # GitHub lists v.202610 first; the release before 202611 is post2
        ("202611", "v.202610.post2"),
        ("202610.post2", "v.202610.post1"),
        ("202610.post1", "v.202610"),
        # -patchN is the old spelling of .postN
        ("202610", "v.202609-patch1"),
        ("202609.post1", "v.202609"),
        ("202605", "v.202604-patch1"),
        ("202402", "v.0.10.6"),
        # a release cut after the newest tag
        ("202612", "v.202611"),
    ],
)
def test_the_previous_release_is_the_highest_version_below(version, previous):
    found = _module().previous_release(_tags(*GITHUB_TAG_ORDER), version)
    assert found.name == previous


@pytest.mark.parametrize(
    "name",
    ["untagged-0cfd089fdcf6777f43a3", "latest", "v.202611rc1", "v.202611.dev1"],
)
def test_a_tag_that_names_no_release_is_never_the_previous_one(name):
    module = _module()
    assert module.tag_version(name) is None
    assert module.previous_release(_tags(name), "202611") is None


def test_the_first_release_has_no_previous_one():
    assert _module().previous_release(_tags("v.202611"), "202611") is None


class _FakeRelease:
    def __init__(self, repo, id, tag_name, title, draft, target_commitish="master"):
        self.repo = repo
        self.id = id
        self.tag_name = tag_name
        self.title = title
        self.draft = draft
        self.target_commitish = target_commitish
        self.html_url = f"https://github.test/releases/{id}"

    def update_release(self, name, message, draft=False, prerelease=False, **kw):
        self.repo.calls.append(("update", dict(kw, name=name, draft=draft)))
        self.title = name
        self.draft = draft
        self.tag_name = kw.get("tag_name", self.tag_name)
        self.target_commitish = kw.get("target_commitish", self.target_commitish)


class _FakeRepo:
    """Enough of a PyGithub Repository for apply_changes.

    `lose_tag` makes the read-back report untagged-<hash>, which is what
    GitHub did with the v.202611 draft.
    """

    def __init__(self, releases=(), lose_tag=False):
        self.releases = []
        self.calls = []
        self.lose_tag = lose_tag
        for tag_name, title, draft in releases:
            self._add(tag_name, title, draft)

    def _add(self, tag_name, title, draft, target_commitish="master"):
        release = _FakeRelease(
            self, len(self.releases) + 1, tag_name, title, draft, target_commitish
        )
        self.releases.append(release)
        return release

    def get_releases(self):
        return list(self.releases)

    def create_git_release(self, tag, name, message, draft=False, **kw):
        self.calls.append(("create", dict(kw, tag=tag, name=name, draft=draft)))
        return self._add(tag, name, draft, **kw)

    def get_release(self, id):
        release = next(r for r in self.releases if r.id == id)
        if self.lose_tag:
            release.tag_name = "untagged-1c1dd7a6e62a636f0b25"
        return release


SHA = "92d65c5a4866c06007c0f36216b592859407e5eb"


def _apply(module, tmp_path, monkeypatch, repo, sha=SHA):
    version_file = tmp_path / "version.txt"
    version_file.write_text("202611\n")
    monkeypatch.setattr(module, "VERSION_FILE", version_file)
    monkeypatch.setattr(module, "update_readme", lambda version: None)
    monkeypatch.setattr(module, "update_pins", lambda version: None)
    monkeypatch.setattr(module, "release_commit", lambda version: sha)
    return module.apply_changes(repo, None, "202611", "notes")


def test_a_new_draft_names_its_tag_and_the_release_commit(tmp_path, monkeypatch):
    module = _module()
    repo = _FakeRepo()
    release = _apply(module, tmp_path, monkeypatch, repo)
    ((call, args),) = repo.calls
    assert call == "create"
    assert args["tag"] == "v.202611"
    assert args["target_commitish"] == SHA
    assert args["draft"] is True
    assert (release.tag_name, release.target_commitish) == ("v.202611", SHA)


def test_a_draft_that_lost_its_tag_is_updated_and_given_it_back(tmp_path, monkeypatch):
    module = _module()
    repo = _FakeRepo([("untagged-1c1dd7a6e62a636f0b25", "ESPnet version 202611", True)])
    release = _apply(module, tmp_path, monkeypatch, repo)
    ((call, args),) = repo.calls
    # not a second draft beside the first
    assert call == "update"
    assert len(repo.releases) == 1
    assert args["tag_name"] == "v.202611"
    assert args["target_commitish"] == SHA
    assert release.tag_name == "v.202611"


def test_a_release_that_reads_back_untagged_stops_the_script(tmp_path, monkeypatch):
    module = _module()
    repo = _FakeRepo(lose_tag=True)
    with pytest.raises(SystemExit, match="untagged-1c1dd7a6e62a636f0b25"):
        _apply(module, tmp_path, monkeypatch, repo)


def test_a_draft_is_not_pinned_before_the_release_commit_exists(tmp_path, monkeypatch):
    module = _module()
    repo = _FakeRepo()
    _apply(module, tmp_path, monkeypatch, repo, sha=None)
    ((_, args),) = repo.calls
    assert "target_commitish" not in args


def test_a_published_release_keeps_its_tag(tmp_path, monkeypatch):
    module = _module()
    repo = _FakeRepo([("v.202611", "ESPnet version 202611", False)])
    _apply(module, tmp_path, monkeypatch, repo)
    ((call, args),) = repo.calls
    assert call == "update"
    assert "tag_name" not in args and "target_commitish" not in args
