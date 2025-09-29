#!/usr/bin/env python3

"""Create release note from milestone with PyGithub.

Launch with:
python doc/make_release_note_from_milestone.py <git_key> \
    <mileston> --llm-ip <llm_ip_address>
"""

import argparse
import json
import sys
from collections import defaultdict

import github
import requests


def make_request(prompt, llm_ip, llm_model, client_type):
    # Make the request to the LLM
    if client_type == "ollama":
        _llm_ip = f"http://{llm_ip}/api/chat"
    else:
        _llm_ip = f"http://{llm_ip}/v1/chat/completions"

    try:
        response = requests.post(
            _llm_ip,
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "model": llm_model,
                "max_tokens": 2048,
                "temperature": 0.7,
                "stream": False,
            },
            headers={"Content-Type": "application/json"},
            timeout=180,
        )

        # Check if the request was successful
        if response.status_code == 200:
            if client_type == "ollama":
                return response.json()["message"]["content"]
            elif client_type == "v1":
                return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Error calling LLM API: {response.status_code}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error generating LLM summary: {e}", file=sys.stderr)
        return None


def generate_llm_summary(
    llm_ip: str,
    llm_model: str,
    client_type: str,
    milestone: str,
    pull_request_dict,
    contributors,
):
    """Generate a summary of the release using an LLM.

    Args:
        llm_ip (str): IP address of the LLM API
        milestone (str): The milestone title
        pull_request_dict (dict): Dictionary containing PRs organized by labels
        contributors (list): List of contributors

    Returns:
        str: The LLM-generated summary
    """

    # Prepare data for the LLM
    pr_data = {}
    for label, prs in pull_request_dict.items():
        pr_data[label] = []
        for pr in prs:
            pr_data[label].append(
                {
                    "title": pr.title,
                    "body": pr.body,
                    "number": pr.number,
                    "author": pr.user.login,
                    "url": pr.html_url,
                }
            )

    # Create the prompt
    prompt = f"""
Generate a comprehensive release note for ESPnet version {milestone}.
Use the following data to create a markdown summary:

PR Data: {json.dumps(pr_data, indent=2)}
Contributors: {json.dumps(contributors, indent=2)}

Format the response as follows:
1. A title section with the milestone name and a brief overview
2. A section for important PRs, highlighting major changes.

The response should be in markdown format.
"""
    return make_request(prompt, llm_ip, llm_model, client_type)


def main():
    """Make release note from milestone with PyGithub."""
    # parse arguments
    parser = argparse.ArgumentParser("release note generator")
    parser.add_argument("--user", default="espnet")
    parser.add_argument("--repo", default="espnet")
    parser.add_argument(
        "--llm-ip",
        default=None,
        help="IP address of the LLM API for generating summary",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-oss:20b",
        help="Model of the LLM API for generating summary",
    )
    parser.add_argument(
        "--llm-type",
        default="ollama",
        choices=["ollama", "v1"],
        help="Model of the LLM API for generating summary",
    )
    parser.add_argument("token", default=None, type=str)
    parser.add_argument("milestone", default=None, type=str)
    args = parser.parse_args()

    # get repository object
    g = github.Github(args.token)
    repo = g.get_organization(args.user).get_repo(args.user)

    # get milestone object
    for m in repo.get_milestones(state="all"):
        if m.title == args.milestone:
            milestone = m
            break

    # get pull requests
    pull_requests = []
    for i in repo.get_issues(milestone, state="closed"):
        try:
            pr = i.as_pull_request()
            if pr.merged:
                pull_requests += [pr]
        except github.UnknownObjectException:
            continue

    # make dict of closed pull requests per label and contributor list
    pull_request_dict = defaultdict(list)
    contributors = []
    pickup_labels = [
        "New Features",
        "Enhancement",
        "Recipe",
        "Bugfix",
        "Documentation",
        "Refactoring",
    ]
    for pr in pull_requests:
        if pr.user.login not in contributors:
            contributors.append(pr.user.login)
        for label in pr.labels:
            is_pickup = False
            if label.name in pickup_labels:
                pull_request_dict[label.name].append(pr)
                is_pickup = True
                break
        if not is_pickup:
            pull_request_dict["Others"].append(pr)

    line_prefix = ""
    # If LLM IP is provided, generate a summary using the LLM
    if args.llm_ip:
        print("# LLM-Generated Summary\n")
        llm_summary = generate_llm_summary(
            args.llm_ip,
            args.llm_model,
            args.llm_type,
            args.milestone,
            pull_request_dict,
            contributors,
        )
        if llm_summary:
            print(llm_summary)
            print("\n## Full changelogn\n")
            line_prefix = "#"
        else:
            print(
                "LLM summary generation failed. "
                "Falling back to standard release note format.\n"
            )

    print(f"\n{line_prefix}# What's Changed\n")
    if args.llm_ip:
        print("\n<details>\n")

    # make release note
    for pickup_label in pickup_labels + ["Others"]:
        if pickup_label not in pull_request_dict:
            continue
        else:
            pull_requests = pull_request_dict[pickup_label]
        print(f"{line_prefix}## {pickup_label}\n")
        for pr in pull_requests:
            line = f"-  {pr.title} (See #{pr.number}, by @{pr.user.login})"
            print(line)
        print()

    if args.llm_ip:
        print("</details>\n\n")

    print(f"{line_prefix}# Acknowledgements")
    print("", end="")
    for idx, contributor in enumerate(sorted(contributors), 1):
        print("@" + contributor, end="")
        if idx != len(contributors):
            print(", ", end="")
        else:
            print(".")
    print()


if __name__ == "__main__":
    main()
