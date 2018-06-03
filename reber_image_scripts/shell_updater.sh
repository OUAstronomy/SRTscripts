#!/bin/bash
_github_="/home/jjtobin/github/SRTscripts/reber_image_scripts/"
_db="${_github_}/live_img.sh"
_dg="${_github_}/srt_win.sh"
_dp="${_github_}/python_updator.sh"
_dr="${_github_}/rt-sun.py"
CWD=`pwd`
while [ true ]; do
  /usr/bin/python ${_dr} -sp
  /usr/bin/bash ${_db} &
  /usr/bin/bash ${_dg} &
  /usr/bin/bash ${_dp} &
  sleep 5
done
