# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checks that the UniGenBench scorer reuses a saved result only for the inputs it was scored from.

Run ``python -m pytest tests/`` from the ``unigenbench`` directory with the scorer's dependencies
installed (pyyaml, tabulate, loguru, tqdm, openai). ``tests/pytest.ini`` keeps the run
self-contained. The judge is a stub, so no VLM endpoint or GPU is used.
"""

import json
import pathlib
import shutil
import sys

import pytest

for _module in ("yaml", "tabulate", "loguru", "tqdm", "openai"):
    pytest.importorskip(_module)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ugb_scorer

TESTPOINTS = ["Style", "Attribute - Color"]
IMAGE_IDS = ["orig0", "orig1", "phi0"]
API_KEY = "secret-api-key"


class StubJudge:
    """Stands in for GatewayVLM: records each construction and passes every testpoint."""

    created: list[str] = []

    def __init__(self, gateway_url, gateway_api, model_str, num_concurrency):
        StubJudge.created.append(model_str)

    def build_request(self, **kwargs):
        return kwargs

    def query(self, requests, pbar_desc=""):
        answer = "\n".join(f'- testpoint: "{p}"\n  analysis: "ok"\n  verdict: true' for p in TESTPOINTS)
        return [answer for _ in requests]

    def close(self):
        pass


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    StubJudge.created = []
    monkeypatch.setattr(ugb_scorer, "GatewayVLM", StubJudge)
    images = tmp_path / "images"
    images.mkdir()
    for image_id in IMAGE_IDS:
        (images / f"{image_id}_0.png").write_bytes(f"image {image_id}".encode())
    rows = [
        {"id": i, "prompt": f"prompt {i}", "sub_dims": {"Testpoints": TESTPOINTS, "Testpoint Description": ["a", "b"]}}
        for i in IMAGE_IDS
    ]
    prompt_json = tmp_path / "prompts.json"
    prompt_json.write_text(json.dumps({"benchmark": rows}))
    return {
        "image_folder": images,
        "benchmark_prompt_json": prompt_json,
        "output_score_json": tmp_path / "score.json",
        "modelstr": "judge-A",
        "gateway_url": "http://judge-a.example/v1",
        "extension": "png",
    }


def run_scorer(monkeypatch, inputs):
    argv = ["ugb_scorer.py", "--gateway_api", API_KEY]
    for key, value in inputs.items():
        argv += [f"--{key}", str(value)]
    monkeypatch.setattr(sys, "argv", argv)
    ugb_scorer.main()


def test_same_inputs_reuse_saved_result(monkeypatch, inputs):
    run_scorer(monkeypatch, inputs)
    saved = inputs["output_score_json"].read_text()
    assert StubJudge.created == ["judge-A"]

    run_scorer(monkeypatch, inputs)

    assert StubJudge.created == ["judge-A"]  # reused, the judge was not called again
    assert inputs["output_score_json"].read_text() == saved
    assert API_KEY not in saved


def _change_judge(inputs, tmp_path, monkeypatch):
    inputs["modelstr"] = "judge-B"


def _edit_prompt_json_in_place(inputs, tmp_path, monkeypatch):
    path = inputs["benchmark_prompt_json"]
    data = json.loads(path.read_text())
    data["benchmark"][0]["prompt"] = "a different prompt"
    path.write_text(json.dumps(data))


def _move_prompt_json(inputs, tmp_path, monkeypatch):
    other = tmp_path / "other_prompts.json"
    shutil.copy(inputs["benchmark_prompt_json"], other)
    inputs["benchmark_prompt_json"] = other


def _regenerate_image_in_place(inputs, tmp_path, monkeypatch):
    (inputs["image_folder"] / "orig0_0.png").write_bytes(b"regenerated image")


def _remove_image(inputs, tmp_path, monkeypatch):
    (inputs["image_folder"] / "phi0_0.png").unlink()


def _change_image_folder(inputs, tmp_path, monkeypatch):
    other = tmp_path / "other_images"
    shutil.copytree(inputs["image_folder"], other)
    inputs["image_folder"] = other


def _missing_image_folder(inputs, tmp_path, monkeypatch):
    inputs["image_folder"] = tmp_path / "does_not_exist"


def _change_extension(inputs, tmp_path, monkeypatch):
    for image in inputs["image_folder"].glob("*.png"):
        shutil.copy(image, image.with_suffix(".jpg"))
    inputs["extension"] = "jpg"


def _change_gateway_url(inputs, tmp_path, monkeypatch):
    inputs["gateway_url"] = "http://judge-b.example/v1"


def _edit_scorer_source(name):
    def change(inputs, tmp_path, monkeypatch):
        # Stands in for an edit to USER_PROMPT, the response parsing or the judge request settings.
        copies = []
        for path in ugb_scorer.SCORER_SOURCE_FILES:
            copy = tmp_path / "edited_scorer" / pathlib.Path(path).name
            copy.parent.mkdir(exist_ok=True)
            shutil.copy(path, copy)
            if copy.name == name:
                copy.write_text(copy.read_text() + "\n# edited\n")
            copies.append(str(copy))
        monkeypatch.setattr(ugb_scorer, "SCORER_SOURCE_FILES", tuple(copies))

    change.__name__ = f"_edit_{name.replace('.', '_')}"
    return change


def _edit_judge_instructions(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(ugb_scorer, "SYSTEM_PROMPT", ugb_scorer.SYSTEM_PROMPT + "\nBe strict.")


@pytest.mark.parametrize(
    "change",
    [
        _change_judge,
        _edit_prompt_json_in_place,
        _move_prompt_json,
        _regenerate_image_in_place,
        _remove_image,
        _change_image_folder,
        _missing_image_folder,
        _change_extension,
        _change_gateway_url,
        _edit_scorer_source("ugb_scorer.py"),
        _edit_scorer_source("utils.py"),
        _edit_judge_instructions,
    ],
    ids=lambda change: change.__name__,
)
def test_different_inputs_do_not_reuse_saved_result(monkeypatch, inputs, tmp_path, change):
    run_scorer(monkeypatch, inputs)
    saved = inputs["output_score_json"].read_text()

    change(inputs, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="scored from different inputs"):
        run_scorer(monkeypatch, inputs)

    assert StubJudge.created == ["judge-A"]
    assert inputs["output_score_json"].read_text() == saved


def test_result_without_run_identity_is_not_reused(monkeypatch, inputs):
    # Result files written before the scorer recorded its inputs cannot be checked.
    run_scorer(monkeypatch, inputs)
    path = inputs["output_score_json"]
    old_result = json.loads(path.read_text())
    old_result.pop("run_identity", None)
    path.write_text(json.dumps(old_result))

    with pytest.raises(SystemExit, match="do not record which inputs"):
        run_scorer(monkeypatch, inputs)

    assert StubJudge.created == ["judge-A"]


def test_notebook_collection_copies_regenerated_images(tmp_path):
    # Rerunning the notebook after regenerating into the same UNIGENBENCH_IMAGE must hand the new
    # images to the scorer, or the image hash in run_identity would still match the old result.
    notebook = pathlib.Path(__file__).resolve().parents[1] / "run_with_cosmos_framework.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    collect = "".join(next(c for c in cells if c.get("id") == "5add5da2-05a4-43c7-a5d5-f9b2b47b1595")["source"])
    (tmp_path / "orig0").mkdir()
    generated = tmp_path / "orig0" / "vision.jpg"

    generated.write_bytes(b"first generation")
    exec(collect, {"Path": pathlib.Path, "UNIGENBENCH_IMAGE": str(tmp_path)})
    generated.write_bytes(b"second generation")
    exec(collect, {"Path": pathlib.Path, "UNIGENBENCH_IMAGE": str(tmp_path)})

    assert (tmp_path / "all_images" / "orig0_0.jpg").read_bytes() == b"second generation"
