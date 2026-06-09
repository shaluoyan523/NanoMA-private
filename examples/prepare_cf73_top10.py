"""Stage the 10 hardest CF-73 problems into a NanoMA workspace."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CF-73 top-10 hardest problems")
    parser.add_argument("--cf73-root", default="/tmp/CF-73", help="Path to cloned ZhouShang0817/CF-73 repository")
    parser.add_argument("--metadata", default="/tmp/cf73_metadata.json", help="Problem metadata JSON from Codeforces API")
    parser.add_argument("--workspace", default="workspace-cf73-top10", help="NanoMA workspace root")
    parser.add_argument(
        "--judge-root",
        default="workspace-cf73-top10-private",
        help="Private evaluator-only root for Polygon packages and hidden judge metadata",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--fetch-statements", action="store_true", help="Fetch Codeforces statements through r.jina.ai")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cf73_root = Path(args.cf73_root)
    workspace = Path(args.workspace)
    judge_root = Path(args.judge_root)
    metadata = json.loads(Path(args.metadata).read_text())
    selected = sorted(metadata, key=lambda r: (-r["rating"], -r["contestId"], r["index"]))[: args.limit]

    target = workspace / "shared" / "cf73_top10"
    private_target = judge_root / "cf73_top10"
    problems_dir = target / "problems"
    private_problems_dir = private_target / "problems"
    shutil.rmtree(target, ignore_errors=True)
    shutil.rmtree(private_target, ignore_errors=True)
    problems_dir.mkdir(parents=True, exist_ok=True)
    private_problems_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    private_manifest = []
    for rank, row in enumerate(selected, start=1):
        problem_id = f"{row['contestId']}{row['index']}"
        package = cf73_root / "polygon_packages" / row["package"]
        problem_dir = problems_dir / problem_id
        private_problem_dir = private_problems_dir / problem_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        private_problem_dir.mkdir(parents=True, exist_ok=True)

        package_info = inspect_polygon_package(package)
        public_item = {
            "rank": rank,
            "problem_id": problem_id,
            "contest_id": row["contestId"],
            "index": row["index"],
            "name": row["name"],
            "rating": row["rating"],
            "tags": row["tags"],
            "url": f"https://codeforces.com/problemset/problem/{row['contestId']}/{row['index']}",
            "short_name": package_info["short_name"],
            "time_limit_ms": package_info["time_limit_ms"],
            "memory_limit_bytes": package_info["memory_limit_bytes"],
            "interactive": package_info["interactive"],
        }
        private_item = {
            **public_item,
            "package": row["package"],
            "test_count": package_info["test_count"],
            "generated_tests": package_info["generated_tests"],
        }
        shutil.copy2(package, private_problem_dir / "polygon_package.zip")
        (problem_dir / "metadata.json").write_text(json.dumps(public_item, indent=2, ensure_ascii=False))
        (private_problem_dir / "metadata.json").write_text(json.dumps(private_item, indent=2, ensure_ascii=False))
        statement = fetch_statement(public_item) if args.fetch_statements else ""
        (problem_dir / "statement.md").write_text(statement or build_statement_stub(public_item))
        manifest.append(public_item)
        private_manifest.append(private_item)

    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (target / "README.md").write_text(build_readme(manifest))
    (private_target / "manifest.json").write_text(json.dumps(private_manifest, indent=2, ensure_ascii=False))
    (private_target / "README.md").write_text(build_private_readme(private_manifest, target))
    print(f"staged {len(manifest)} problems under {target}")
    print(f"private judge material under {private_target}")


def inspect_polygon_package(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        xml = ET.fromstring(zf.read("problem.xml"))
        testset = xml.find("./judging/testset")
        tests = testset.findall("./tests/test") if testset is not None else []
        time_limit = int(testset.findtext("time-limit", "0")) if testset is not None else 0
        memory_limit = int(testset.findtext("memory-limit", "0")) if testset is not None else 0
        interactive = xml.find("./assets/interactor") is not None
        short_name = xml.attrib.get("short-name", "")
        generated = []
        for test in tests:
            generated.append({"cmd": test.attrib.get("cmd", ""), "method": test.attrib.get("method", "")})
    return {
        "short_name": short_name,
        "time_limit_ms": time_limit,
        "memory_limit_bytes": memory_limit,
        "test_count": len(generated),
        "interactive": interactive,
        "generated_tests": generated,
    }


def build_statement_stub(item: dict) -> str:
    tags = ", ".join(item["tags"])
    inter = "yes" if item["interactive"] else "no"
    return f"""# {item['problem_id']} - {item['name']}

Rating: {item['rating']}
Tags: {tags}
URL: {item['url']}

The public Codeforces statement could not be fetched while staging. Fetch or paste the
public statement before running a scored agent attempt.

## Public Metadata

- Time limit: {item['time_limit_ms']} ms
- Memory limit: {item['memory_limit_bytes']} bytes
- Interactive: {inter}
"""


def fetch_statement(item: dict) -> str:
    url = f"https://r.jina.ai/http://r.jina.ai/http://https://codeforces.com/problemset/problem/{item['contest_id']}/{item['index']}?locale=en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"warning: failed to fetch statement for {item['problem_id']}: {e}")
        return ""
    if "Markdown Content:" not in text or "Just a moment" in text:
        print(f"warning: fetched statement for {item['problem_id']} did not look usable")
        return ""
    text = text.replace("Markdown Content:", "## Statement\n", 1)
    return f"# {item['problem_id']} - {item['name']}\n\nRating: {item['rating']}\nURL: {item['url']}\n\n{text.strip()}\n"


def build_readme(manifest: list[dict]) -> str:
    lines = [
        "# CF-73 Top-10 Hardest",
        "",
        "Selected by official Codeforces rating from the released CF-73 Polygon packages.",
        "This public workspace intentionally excludes Polygon packages, official solutions, checkers, interactors, hidden test metadata, and generated-test commands.",
        "",
        "| Rank | Problem | Rating | Name | Interactive |",
        "|---:|---|---:|---|---|",
    ]
    for item in manifest:
        lines.append(
            f"| {item['rank']} | {item['problem_id']} | {item['rating']} | "
            f"{item['name']} | {item['interactive']} |"
        )
    lines.extend(
        [
            "",
            "Agents should use only `statement.md` and public `metadata.json` files in this tree.",
            "Private judge material lives outside the agent workspace and is used only after a final solution is selected.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_private_readme(manifest: list[dict], public_target: Path) -> str:
    lines = [
        "# CF-73 Private Judge Material",
        "",
        "Evaluator-only copy of Polygon packages and hidden generated-test metadata.",
        f"Public agent-visible workspace: `{public_target}`",
        "",
        "| Rank | Problem | Package | Tests | Interactive |",
        "|---:|---|---|---:|---|",
    ]
    for item in manifest:
        lines.append(
            f"| {item['rank']} | {item['problem_id']} | {item['package']} | "
            f"{item['test_count']} | {item['interactive']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
