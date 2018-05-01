#!/usr/bin/bash
# Created by Nickalas Reynolds
# Autoupdates the latest files

# live feed camera of srt
readonly IMG_TEMP_ORIG_PATH_='/home/jjtobin/public_html/camera_images'
readonly IMG_TEMP_FINAL_PATH_='/home/jjtobin/public_html/img'

readonly IMG_TEMP_NAME_=`ls -t $IMG_TEMP_ORIG_PATH_ | head -n1`
readonly IMG_TEMP_NEWNAME_='curr_live_img.jpg'

readonly IMG_F_TEMP_NAME_="$IMG_TEMP_ORIG_PATH_"'/'"$IMG_TEMP_NAME_"
readonly IMG_F_TEMP_NEWNAME_="$IMG_TEMP_FINAL_PATH_"'/'"$IMG_TEMP_NEWNAME_"

cp -f $IMG_F_TEMP_NAME_ $IMG_F_TEMP_NEWNAME_

# live update of srtn program
readonly TER_TEMP_ORIG_PATH_='/home/jjtobin/public_html/window_images'
readonly TER_TEMP_FINAL_PATH_='/home/jjtobin/public_html/img'

readonly TER_TEMP_NAME_=`ls -t $TER_TEMP_ORIG_PATH_ | head -n1`
readonly TER_TEMP_NEWNAME_='curr_terminal.jpg'

readonly TER_F_TEMP_NAME_="$TER_TEMP_ORIG_PATH_"'/'"$TER_TEMP_NAME_"
readonly TER_F_TEMP_NEWNAME_="$TER_TEMP_FINAL_PATH_"'/'"$TER_TEMP_NEWNAME_"

cp -f $TER_F_TEMP_NAME_ $TER_F_TEMP_NEWNAME_

# live updated python integrated flux
readonly PYT_TEMP_ORIG_PATH_='/home/jjtobin/public_html/python_integrated'
readonly PYT_TEMP_FINAL_PATH_='/home/jjtobin/public_html/img'

readonly PYT_TEMP_NAME_=`ls -t $PYT_TEMP_ORIG_PATH_ | head -n1`
readonly PYT_TEMP_NEWNAME_='realsun_integ.png'

readonly PYT_F_TEMP_NAME_="$PYT_TEMP_ORIG_PATH_"'/'"$PYT_TEMP_NAME_"
readonly PYT_F_TEMP_NEWNAME_="$PYT_TEMP_FINAL_PATH_"'/'"$PYT_TEMP_NEWNAME_"

cp -f $PYT_F_TEMP_NAME_ $PYT_F_TEMP_NEWNAME_
