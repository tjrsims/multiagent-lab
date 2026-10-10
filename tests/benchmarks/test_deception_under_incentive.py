"""Benchmark 1 as a module: the base stays pure, the module runs end to end offline."""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from benchmarks.deception_under_incentive import PRESETS, build_module
from benchmarks.deception_under_incentive.analysis import summarize
from benchmarks.deception_under_incentive.report import render_leaderboard
from end_bench.cli import cli
from end_bench.data.text_dataset import load_text_dataset
from end_bench.run import RunSpecError, run_text_track
from interpretability.runtime.runner import builtin_label_track_registry
from interpretability.scenarios.compiled import compile_emergent_scenario, render_counterpart_prompt
from interpretability.scenarios.policy_registry import builtin_policy_registry

REPO = Path(__file__).resolve().parents[2]


# T4.1 importing and building the module changes nothing in the base.

def test_base_packages_never_import_benchmarks() -> None:
    offenders = []
    for package in ("interpretability", "negotiation", "end_bench", "config"):
        for path in (REPO / package).rglob("*.py"):
            for line in path.read_text().splitlines():
                if re.match(r"^\s*(from|import)\s+benchmarks\b", line):
                    offenders.append(f"{path.relative_to(REPO)}: {line.strip()}")
    assert offenders == []


def test_building_the_module_leaves_the_builtin_registries_untouched() -> None:
    before_policies = builtin_policy_registry().names()
    before_tracks = builtin_label_track_registry().bindings()
    module = build_module()
    assert "interrogative" in module.policy_registry
    assert builtin_policy_registry().names() == before_policies
    assert builtin_label_track_registry().bindings() == before_tracks
    assert build_module().policy_registry is not module.policy_registry  # fresh objects, no globals


# T4.3 the questioning policy reaches the counterpart prompt in every scenario.

@pytest.mark.parametrize("scenario", ["hidden_value", "capability_bluff", "info_withholding", "ultimatum_bluff"])
def test_interrogative_policy_renders_a_question_instruction(scenario: str) -> None:
    module = build_module()
    instance = compile_emergent_scenario(
        scenario, family_seed=3, trial_seed=1, condition="high_incentive",
        role_assignment={"actor": "Negotiator", "counterpart": "Counterpart"},
        counterpart_type="interrogative", policy_registry=module.policy_registry,
    )
    prompt = render_counterpart_prompt(instance, "Counterpart", module.policy_registry)
    assert "ask the other participant one direct question" in prompt
    assert "lie" not in prompt.lower().split("counterpart policy")[-1]


# T4.2, T4.4, T4.5 an offline run through the module, then analysis and the leaderboard.

@pytest.fixture(scope="module")
def module_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("module_run")
    spec = dataclasses.replace(
        PRESETS["pilot"]("scripted", counterpart_type="interrogative", run_id="mod-smoke"),
        scenarios=("hidden_value", "ultimatum_bluff"), families=2, max_rounds=3,
    )
    return run_text_track(spec, out, module=build_module()), out


def test_module_run_records_provenance_on_manifest_and_rows(module_run) -> None:
    result, out = module_run
    module = build_module()
    manifest = json.loads((out / "mod-smoke" / "run_manifest.json").read_text())
    assert manifest["benchmark_module"]["name"] == "deception_under_incentive"
    assert manifest["benchmark_module"]["version"] == "0.1.0"
    assert manifest["closing_attestation"] is True
    assert manifest["closing_attestation_source"] == "registered:closing-attestation/2"
    assert manifest["spec"]["closing_attestation"] is True
    assert manifest["execution_plan"]["counterpart_policy_ids"] == [module.policy_registry.get("interrogative").policy_id]
    dataset = load_text_dataset(result.text_dataset_path)
    rows = dataset["metadata"]
    assert rows
    assert {r["counterpart_type"] for r in rows} == {"interrogative"}
    assert {r["counterpart_policy_id"] for r in rows} == {module.policy_registry.get("interrogative").policy_id}
    attested = [r for r in rows if r.get("intervention_application_receipt_ids")]
    assert attested, "the closing attestation must leave a receipt on at least one row"


def test_module_analysis_and_leaderboard_run_on_the_smoke_dataset(module_run) -> None:
    result, out = module_run
    dataset = load_text_dataset(result.text_dataset_path)
    summary = summarize(dataset, n_bootstrap=50)
    assert summary["n_rows"] > 0
    cells = summary["cells"]
    assert {c["scenario"] for c in cells} == {"hidden_value", "ultimatum_bluff"}
    assert all(c["counterpart_type"] == "interrogative" for c in cells)
    assert any(c["attestation_coverage"] for c in cells)
    table = render_leaderboard([out / "mod-smoke"])
    assert "| mod-smoke | scripted | interrogative |" in table


def test_module_rejects_registries_passed_beside_it(tmp_path) -> None:
    spec = PRESETS["pilot"]("scripted", counterpart_type="default", run_id="dup")
    with pytest.raises(RunSpecError, match="not both"):
        run_text_track(spec, tmp_path, module=build_module(), policy_registry=builtin_policy_registry())


def test_cli_module_run_and_leaderboard(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [
        "module", "run", "deception_under_incentive", "--model", "scripted", "--preset", "pilot",
        "--counterpart-type", "default", "--run-id", "cli", "--out", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "deception_under_incentive 0.1.0 preset=pilot policy=default" in result.output
    run_dir = tmp_path / "cli-pilot-default"
    assert (run_dir / "text_dataset.json").exists()
    board = runner.invoke(cli, ["module", "leaderboard", "deception_under_incentive", str(run_dir)])
    assert board.exit_code == 0, board.output
    assert "| cli-pilot-default | scripted | default |" in board.output
    pilot = runner.invoke(cli, ["module", "pilot", "deception_under_incentive", str(run_dir),
                                "--out", str(tmp_path / "pilot"), "--n-bootstrap", "20"])
    assert pilot.exit_code == 0, pilot.output
    assert (tmp_path / "pilot" / "PILOT_REPORT.md").exists() and (tmp_path / "pilot" / "pilot_cells.json").exists()
    assert "Dropped cells" in pilot.output
    missing = runner.invoke(cli, ["module", "run", "nope", "--model", "scripted"])
    assert missing.exit_code != 0 and "no benchmark module named 'nope'" in missing.output


def test_pilot_table_and_cell_selection(module_run) -> None:
    from benchmarks.deception_under_incentive.pilot import cell_table, render_pilot_report, select_cells

    _, out = module_run
    table = cell_table([out / "mod-smoke"], n_bootstrap=20)
    assert {r["scenario"] for r in table} == {"hidden_value", "ultimatum_bluff"}
    assert all(r["model"] == "scripted" for r in table)
    selection = select_cells(table)
    assert set(selection) == {"kept", "dropped", "min_trial_coverage", "min_lie_rate"}
    assert len(selection["kept"]) + len(selection["dropped"]) == 2
    for item in selection["dropped"]:
        assert item["reasons"], "every dropped cell says why"
    assert all(0.0 <= r["trial_claim_coverage"] <= 1.0 for r in table)
    strict = select_cells(table, min_trial_coverage=1.01)
    assert not strict["kept"] and all("trials hold a checkable claim" in r for d in strict["dropped"] for r in d["reasons"])
    report = render_pilot_report(table, selection)
    assert "| scripted | hidden_value | interrogative |" in report and "Dropped cells" in report



def test_manifest_records_the_preregistration_hash() -> None:
    import hashlib

    from benchmarks.deception_under_incentive.manifest import PREREG_PATH, prereg_hash

    module = build_module()
    assert PREREG_PATH.exists()
    assert module.prereg_sha256 == prereg_hash() == hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    assert module.provenance()["prereg_sha256"] == module.prereg_sha256



def test_cli_module_run_finishes_the_batch_when_one_policy_fails(tmp_path, monkeypatch) -> None:
    import end_bench.cli as cli_module
    from end_bench.run import RunSpecError

    calls = []

    def fake_run(spec, out_dir, *, module=None, **_kwargs):
        calls.append(spec.counterpart_types[0])
        if spec.counterpart_types[0] == "default":
            raise RunSpecError("preflight call failed: EndpointConnectionError")
        return type("R", (), {"text_dataset_path": tmp_path / "x", "run_dir": tmp_path, "n_rows": 1})()

    monkeypatch.setattr("end_bench.run.run_text_track", fake_run)
    monkeypatch.setattr(cli_module, "_echo_result", lambda result: None)
    result = CliRunner().invoke(cli, [
        "module", "run", "deception_under_incentive", "--model", "scripted", "--preset", "pilot",
        "--counterpart-type", "default", "--counterpart-type", "skeptical", "--run-id", "b", "--out", str(tmp_path),
    ])
    assert calls == ["default", "skeptical"], "the batch continued past the failure"
    assert result.exit_code != 0
    assert "1 of 2 policy runs failed" in result.output and "EndpointConnectionError" in result.output



def test_presets_scale_as_documented() -> None:
    pilot = PRESETS["pilot"]("scripted")
    lite = PRESETS["main_lite"]("scripted")
    main = PRESETS["main"]("scripted")
    assert pilot.families == 3 and pilot.conditions == ("high_incentive",) and pilot.rollouts_per_family == 1
    assert lite.families == 30 and len(lite.conditions) == 3 and lite.rollouts_per_family == 1
    assert main.families == 30 and len(main.conditions) == 3 and main.rollouts_per_family == 5
    assert all(spec.paired_facts and spec.closing_attestation for spec in (pilot, lite, main))
