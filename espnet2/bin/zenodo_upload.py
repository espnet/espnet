"""Upload files to Zenodo.

You need to do as follows in order to access zenodo:

1. Sign up to Zenodo: https://zenodo.org/
2. Create access_token: https://zenodo.org/account/settings/applications/tokens/new/
"""

import argparse
from datetime import datetime
from getpass import getpass
import json
import os
from pathlib import Path
import requests
from typing import Collection
from typing import Union

from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool


class Zenodo:
    """Helper class to invoke Zenodo API

    REST API of zenodo: https://developers.zenodo.org/

    """

    def __init__(self, access_token: str, use_sandbox: bool = False):
        if use_sandbox:
            self.zenodo_url = "https://sandbox.zenodo.org"
        else:
            self.zenodo_url = "https://zenodo.org"

        self.params = {"access_token": access_token}
        self.headers = {"Content-Type": "application/json"}

    def create_deposit(self) -> requests.models.Response:
        r = requests.post(
            f"{self.zenodo_url}/api/deposit/depositions",
            params=self.params,
            json={},
            headers=self.headers,
        )
        if r.status_code != 201:
            raise RuntimeError(r.json()["message"])
        return r

    def update_metadata(
        self, r: Union[requests.models.Response, int], data
    ) -> requests.models.Response:
        if isinstance(r, requests.models.Response):
            deposition_id = r.json()["id"]
        else:
            deposition_id = r

        r = requests.put(
            f"{self.zenodo_url}/api/deposit/depositions/{deposition_id}",
            params=self.params,
            data=json.dumps(data),
            headers=self.headers,
        )
        if r.status_code != 200:
            raise RuntimeError(r.json()["message"])
        return r

    def upload_file(
        self, r: Union[requests.models.Response, int], filename: Union[Path, str]
    ) -> requests.models.Response:
        if isinstance(r, int):
            r = requests.get(
                f"{self.zenodo_url}/api/deposit/depositions/{r}", headers=self.headers
            )

        bucket_url = r.json()["links"]["bucket"]
        name = Path(filename).name
        with open(filename, "rb") as fp:
            r = requests.put(
                f"{bucket_url}/{name}",
                data=fp,
                # No headers included since it's a raw byte request
                params=self.params,
            )
            if r.status_code != 200:
                raise RuntimeError(r.json()["message"])
        return r

    def publish(
        self, r: Union[requests.models.Response, int]
    ) -> requests.models.Response:
        if isinstance(r, requests.models.Response):
            deposition_id = r.json()["id"]
        else:
            deposition_id = r

        r = requests.post(
            f"{self.zenodo_url}/api/deposit/depositions/"
            f"{deposition_id}/actions/publish",
            params=self.params,
        )
        if r.status_code != 202:
            raise RuntimeError(r.json()["message"])
        return r


def upload(
    access_token: str,
    title: str,
    creator_name: str,
    description: str = "",
    files: Collection[Union[Path, str]] = (),
    affiliation: str = None,
    orcid: str = None,
    gnd: str = None,
    upload_type: str = "other",
    license: str = "CC-BY-4.0",
    keywords: Collection[str] = (),
    related_identifiers: Collection[dict] = (),
    community_identifer: str = "espnet",
    use_sandbox: bool = True,
    publish: bool = False,
):
    zenodo = Zenodo(access_token, use_sandbox=use_sandbox)
    r = zenodo.create_deposit()

    # Update metatdata using old API
    creator = {"name": creator_name}
    if affiliation is not None:
        creator["affiliation"] = affiliation
    if orcid is not None:
        creator["orcid"] = orcid
    if gnd is not None:
        creator["gnd"] = gnd
    data = {
        "metadata": {
            "upload_type": upload_type,
            "publication_date": datetime.now().strftime("%Y-%m-%d"),
            "title": title,
            "description": description,
            "creators": [creator],
            "communities": [{"identifier": community_identifer}],
            "license": license,
            "keywords": list(keywords),
            "related_identifiers": list(related_identifiers),
        }
    }
    zenodo.update_metadata(r, data)

    # Upload files using new API
    for f in files:
        # Check file existing
        if not Path(f).exists():
            raise FileNotFoundError(f"{f} is not found")
    for f in files:
        print(f"Now uploading {f}...")
        zenodo.upload_file(r, f)

    if publish:
        r = zenodo.publish(r)
        url = r.json()["links"]["latest_html"]
        print(f"Successfully published. Go to {url}")
    else:
        url = r.json()["links"]["html"]
        print(f"Successfully uploaded, but not published yet. Go to {url}")


def upload_espnet_model(
    access_token: str,
    title: str,
    creator_name: str,
    file: Collection[Union[Path, str]] = (),
    description: str = "",
    description_file: str = None,
    affiliation: str = None,
    license: str = "CC-BY-4.0",
    orcid: str = None,
    gnd: str = None,
    use_sandbox: bool = False,
    publish: bool = False,
):
    if description_file is not None:
        with open(description_file, "r", encoding="utf-8") as f:
            description = f.read()

    upload(
        access_token=access_token,
        title=title,
        description=description,
        creator_name=creator_name,
        files=file,
        keywords=[
            "ESPnet",
            "deep-learning",
            "python",
            "pytorch",
            "speech-recognition",
            "speech-synthesis",
            "speech-translation",
            "machine-translation",
        ],
        related_identifiers=[
            {
                "relation": "isSupplementTo",
                "identifier": "https://github.com/espnet/espnet",
            }
        ],
        affiliation=affiliation,
        license=license,
        orcid=orcid,
        gnd=gnd,
        use_sandbox=use_sandbox,
        publish=publish,
    )


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Upload files to Zenodo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--access_token",
        help="Get your access_token from "
        "https://zenodo.org/account/settings/applications/ or  "
        "https://sandbox.zenodo.org/account/settings/applications/ . "
        "You can also give it from an environment variable 'ACCESS_TOKEN'",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="e.g. ESPnet pretrained model, MT, "
        "Fisher-CallHome Spanish (Es->En), Transformer",
    )
    parser.add_argument("--creator_name", required=True, help="Your name")
    parser.add_argument("--file", nargs="+", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--description", help="Give the description")
    group.add_argument("--description_file", help="Give the description from file")
    parser.add_argument(
        "--use_sandbox",
        type=str2bool,
        default=False,
        help="Use zenodo sandbox for testing",
    )
    parser.add_argument(
        "--publish", type=str2bool, default=False, help="Publish after uploading"
    )
    parser.add_argument("--license", default="CC-BY-4.0")
    parser.add_argument("--affiliation")
    parser.add_argument("--orcid")
    parser.add_argument("--gnd")
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)

    # If --access_token is not given, get from "ACCESS_TOKEN"
    if args.access_token is None:
        args.access_token = os.environ.get("ACCESS_TOKEN")

    # If neither is given, input from stdin
    if args.access_token is None:
        if args.use_sandbox:
            zenodo_url = "https://sandbox.zenodo.org"
        else:
            zenodo_url = "https://zenodo.org"
        args.access_token = getpass(
            "Input Zenodo API Token\n"
            "(You can create it from "
            f"{zenodo_url}/account/settings/applications/tokens/new/): "
        )

    kwargs = vars(args)
    kwargs.pop("config")
    upload_espnet_model(**kwargs)


if __name__ == "__main__":
    main()
