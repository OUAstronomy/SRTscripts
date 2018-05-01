#!/bin/bash
_github_=""
_db="${_github_}/live_img.sh"
_dg="${_github_}/srt_win.sh"
_dp="${_github_}/python_updator.sh"
_dy="/home/jjtobin/RealTime-Sun/"
CWD=`pwd`
while [ true ]; do
  cd "${_dy}"
  /usr/bin/env python rt-sun.py
  cd "$CWD"
  /usr/bin/bash ${_db} &
  /usr/bin/bash ${_dg} &
  /usr/bin/bash ${_dp} &
  sleep 5
done
