#!/usr/bin/env python3

"""Create release note from milestone with PyGithub."""

import argparse

from collections import defaultdict

import github


def main():
    """Make release note from milestone with PyGithub."""
    # parse arguments
    parser = argparse.ArgumentParser("release note generator")
    parser.add_argument("--user", default="espnet")
    parser.add_argument("--repo", default="espnet")
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

    # make release note
    ignore_labels = ["mergify", "auto-merge", "README"]
    for pickup_label in pickup_labels + ["Others"]:
        if pickup_label not in pull_request_dict:
            continue
        else:
            pull_requests = pull_request_dict[pickup_label]
        print(f"# {pickup_label}")
        for pr in pull_requests:
            sub_labels = [
                l_.name
                for l_ in pr.labels
                if l_.name not in pickup_labels + ignore_labels
            ]
            line = f"- [**{pickup_label}**]"
            sub_line_a = ""
            sub_line_b = ""
            for sub_label in sub_labels:
                if sub_label.startswith("ESPnet"):
                    sub_line_a += f"[**{sub_label}**]"
                else:
                    sub_line_b += f"[**{sub_label}**]"
            line += sub_line_a + sub_line_b
            line += f" {pr.title} #{pr.number} by @{pr.user.login}"
            print(line)
        print()

    print("# Acknowledgements")
    print("Special thanks to ", end="")
    for idx, contributor in enumerate(sorted(contributors), 1):
        print("@" + contributor, end="")
        if idx != len(contributors):
            print(", ", end="")
        else:
            print(".")
    print()


if __name__ == "__main__":
    main()
