"""Static checks on the Kubernetes manifests: the mistakes that only show up during an incident."""

import sys
from pathlib import Path

import pytest
import yaml

CODE = Path(__file__).resolve().parents[1]
K8S = CODE / "k8s"
sys.path.insert(0, str(CODE / "services" / "chat"))

from ch20_chat.core import load_prompts
from ch20_chat.settings import load_settings


def docs(*paths):
    out = []
    for p in paths:
        out += [d for d in yaml.safe_load_all(p.read_text()) if d]
    return out


BASE = docs(*sorted(p for p in (K8S / "base").glob("*.yaml")
                    if p.name != "kustomization.yaml"))
BY_KIND: dict[str, dict[str, dict]] = {}
for d in BASE:
    BY_KIND.setdefault(d["kind"], {})[d["metadata"]["name"]] = d
DEPLOYMENTS = BY_KIND["Deployment"]


def containers(dep):
    return dep["spec"]["template"]["spec"]["containers"]


@pytest.mark.parametrize("name", sorted(DEPLOYMENTS))
def test_every_container_has_three_probes_and_liveness_skips_dependencies(name):
    for c in containers(DEPLOYMENTS[name]):
        assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(c)
        assert c["livenessProbe"]["httpGet"]["path"] == "/healthz"
        assert c["readinessProbe"]["httpGet"]["path"] == "/readyz"


@pytest.mark.parametrize("name", sorted(DEPLOYMENTS))
def test_requests_limits_and_pinned_images(name):
    for c in containers(DEPLOYMENTS[name]):
        res = c["resources"]
        assert {"cpu", "memory"} <= set(res["requests"])     # HPA utilization needs the cpu request
        assert "memory" in res["limits"]
        repo, _, tag = c["image"].rpartition(":")
        assert repo and tag and tag != "latest"
        assert c["imagePullPolicy"] == "IfNotPresent"         # kind: images are side-loaded


@pytest.mark.parametrize("name", sorted(DEPLOYMENTS))
def test_non_root_and_read_only(name):
    pod = DEPLOYMENTS[name]["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    for c in containers(DEPLOYMENTS[name]):
        assert c["securityContext"]["readOnlyRootFilesystem"] is True
        assert c["securityContext"]["allowPrivilegeEscalation"] is False


@pytest.mark.parametrize("name", sorted(DEPLOYMENTS))
def test_graceful_shutdown_fits_in_the_grace_period(name):
    spec = DEPLOYMENTS[name]["spec"]
    grace = spec["template"]["spec"]["terminationGracePeriodSeconds"]
    for c in containers(DEPLOYMENTS[name]):
        pre_stop = c["lifecycle"]["preStop"]["sleep"]["seconds"]
        assert 5 <= pre_stop < grace
    assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0


def test_services_select_their_deployments_and_named_ports_exist():
    for name, svc in BY_KIND["Service"].items():
        if "selector" not in svc["spec"]:
            continue                                    # external service (shopflow-db)
        dep = DEPLOYMENTS[name]
        labels = dep["spec"]["template"]["metadata"]["labels"]
        assert svc["spec"]["selector"].items() <= labels.items()
        port_names = {p.get("name") for c in containers(dep) for p in c["ports"]}
        for p in svc["spec"]["ports"]:
            assert p["targetPort"] in port_names


def test_pdbs_select_pods_and_never_block_every_drain():
    hpas = {h["spec"]["scaleTargetRef"]["name"]: h
            for h in BY_KIND["HorizontalPodAutoscaler"].values()}
    for name, pdb in BY_KIND["PodDisruptionBudget"].items():
        dep = DEPLOYMENTS[name]
        labels = dep["spec"]["template"]["metadata"]["labels"]
        assert pdb["spec"]["selector"]["matchLabels"].items() <= labels.items()
        replicas = dep["spec"]["replicas"]
        min_replicas = hpas[name]["spec"]["minReplicas"] if name in hpas else replicas
        if "minAvailable" in pdb["spec"]:
            # minAvailable == replicas would make every node drain hang forever.
            assert pdb["spec"]["minAvailable"] < min_replicas
        else:
            assert pdb["spec"]["maxUnavailable"] >= 1


def test_hpa_targets_an_existing_deployment_with_sane_bounds():
    for hpa in BY_KIND["HorizontalPodAutoscaler"].values():
        assert hpa["spec"]["scaleTargetRef"]["name"] in DEPLOYMENTS
        assert 2 <= hpa["spec"]["minReplicas"] < hpa["spec"]["maxReplicas"]


def test_chat_configmap_is_loadable_and_names_real_prompt_versions():
    env = dict(line.split("=", 1) for line in (K8S / "base/chat.env").read_text().splitlines()
               if line and not line.startswith("#"))
    secrets = {"LLM_API_KEY": "from-secret", "SESSIONS_DATABASE_URL": "postgresql://x"}
    s = load_settings({**env, **secrets})
    prompts = load_prompts()
    assert s.prompt_stable in prompts and s.prompt_canary in prompts


def test_kind_port_mappings_match_service_node_ports():
    kind = yaml.safe_load((K8S / "kind-config.yaml").read_text())
    mapped = {m["containerPort"] for n in kind["nodes"] for m in n.get("extraPortMappings", [])}
    node_ports = {p["nodePort"] for s in BY_KIND["Service"].values()
                  for p in s["spec"]["ports"] if "nodePort" in p}
    assert node_ports <= mapped


def test_kustomization_lists_every_base_file():
    k = yaml.safe_load((K8S / "base/kustomization.yaml").read_text())
    files = {p.name for p in (K8S / "base").glob("*.yaml")} - {"kustomization.yaml"}
    assert set(k["resources"]) == files
