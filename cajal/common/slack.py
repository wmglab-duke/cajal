import os
import requests
import json

from cajal.mpi import Backend as MPI


TOKEN = os.getenv("SLACK_TOKEN")
CHANNEL = "#cluster-jobs"
ICON = "https://rc.dartmouth.edu/wp-content/uploads/2016/12/discovery_icon_20161208_forweb_new150x150.png"
USERNAME = "Cluster Monitor"


def post_message_to_slack(text, icon=False, token=None, channel=None, blocks=None):
    if MPI.MASTER():
        ret = requests.post(
            "https://slack.com/api/chat.postMessage",
            {
                "token": token or TOKEN,
                "channel": channel or CHANNEL,
                "text": text,
                "icon_url": ICON if icon else None,
                "username": USERNAME,
                "blocks": json.dumps(blocks) if blocks else None,
            },
        ).json()
    else:
        ret = None
    return ret
